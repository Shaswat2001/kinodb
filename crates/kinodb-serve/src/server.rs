//! kinodb gRPC server — serves trajectory data to training workers.
//!
//! Usage:
//!   kino-serve data.kdb --port 50051
//!   kino-serve --source bridge.kdb:0.4 --source aloha.kdb:0.6 --port 50051

use std::sync::{Arc, Mutex};

use kinodb_core::{kql, KdbReader, Mixture};
use tonic::{Request, Response, Status};

pub mod pb {
    tonic::include_proto!("kinodb");
}

use pb::kino_service_server::KinoService;

// ── Data backend ────────────────────────────────────────────

/// The server can serve from a single .kdb file or a weighted mixture.
enum Backend {
    Single {
        reader: KdbReader,
        path: String,
    },
    Mix {
        mixture: Mutex<Mixture>,
        sources: Vec<String>,
    },
}

impl Backend {
    fn num_episodes(&self) -> usize {
        match self {
            Backend::Single { reader, .. } => reader.num_episodes(),
            Backend::Mix { mixture, .. } => mixture.lock().unwrap().total_episodes(),
        }
    }

    fn num_frames(&self) -> u64 {
        match self {
            Backend::Single { reader, .. } => reader.num_frames(),
            Backend::Mix { mixture, .. } => mixture.lock().unwrap().total_frames(),
        }
    }

    fn read_episode(&self, pos: usize) -> Result<kinodb_core::Episode, Status> {
        match self {
            Backend::Single { reader, .. } => reader
                .read_episode(pos)
                .map_err(|e| Status::not_found(format!("episode {}: {}", pos, e))),
            Backend::Mix { mixture, .. } => mixture
                .lock()
                .unwrap()
                .read_global(pos)
                .map_err(|e| Status::not_found(format!("global {}: {}", pos, e))),
        }
    }

    fn read_meta(&self, pos: usize) -> Result<kinodb_core::EpisodeMeta, Status> {
        match self {
            Backend::Single { reader, .. } => reader
                .read_meta(pos)
                .map_err(|e| Status::not_found(format!("meta {}: {}", pos, e))),
            Backend::Mix { mixture, .. } => {
                // Read full episode and return meta (mixture doesn't have read_global_meta)
                let ep = mixture
                    .lock()
                    .unwrap()
                    .read_global(pos)
                    .map_err(|e| Status::not_found(format!("global meta {}: {}", pos, e)))?;
                Ok(ep.meta)
            }
        }
    }

    fn sample(&self) -> Result<kinodb_core::Episode, Status> {
        match self {
            Backend::Single { reader, .. } => {
                // Simple random using position
                let n = reader.num_episodes();
                if n == 0 {
                    return Err(Status::not_found("empty database"));
                }
                let pos = (std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as usize)
                    % n;
                reader
                    .read_episode(pos)
                    .map_err(|e| Status::internal(format!("{}", e)))
            }
            Backend::Mix { mixture, .. } => mixture
                .lock()
                .unwrap()
                .sample()
                .map_err(|e| Status::internal(format!("{}", e))),
        }
    }

    fn num_sources(&self) -> u32 {
        match self {
            Backend::Single { .. } => 1,
            Backend::Mix { sources, .. } => sources.len() as u32,
        }
    }

    fn source_names(&self) -> Vec<String> {
        match self {
            Backend::Single { path, .. } => vec![path.clone()],
            Backend::Mix { sources, .. } => sources.clone(),
        }
    }
}

// ── gRPC service ────────────────────────────────────────────

pub struct KinoServer {
    backend: Arc<Backend>,
}

impl KinoServer {
    pub fn from_single(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let reader = KdbReader::open(path)?;
        Ok(Self {
            backend: Arc::new(Backend::Single {
                reader,
                path: path.to_string(),
            }),
        })
    }

    pub fn from_mixture(
        sources: &[(String, f64)],
        seed: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut builder = Mixture::builder().seed(seed);
        let mut source_names = Vec::new();
        for (path, weight) in sources {
            builder = builder.add(path.as_str(), *weight);
            source_names.push(path.clone());
        }
        let mixture = builder.build()?;
        Ok(Self {
            backend: Arc::new(Backend::Mix {
                mixture: Mutex::new(mixture),
                sources: source_names,
            }),
        })
    }
}

// ── Conversions ─────────────────────────────────────────────

fn meta_to_pb(m: &kinodb_core::EpisodeMeta) -> pb::EpisodeMeta {
    pb::EpisodeMeta {
        episode_id: m.id.0,
        embodiment: m.embodiment.clone(),
        task: m.language_instruction.clone(),
        num_frames: m.num_frames,
        fps: m.fps,
        action_dim: m.action_dim as u32,
        has_success: m.success.is_some(),
        success: m.success.unwrap_or(false),
        has_reward: m.total_reward.is_some(),
        total_reward: m.total_reward.unwrap_or(0.0),
    }
}

fn frame_to_pb(f: &kinodb_core::Frame, include_images: bool) -> pb::Frame {
    let images = if include_images {
        f.images
            .iter()
            .map(|img| pb::ImageObs {
                camera: img.camera.clone(),
                width: img.width,
                height: img.height,
                channels: img.channels as u32,
                data: img.data.clone(),
            })
            .collect()
    } else {
        vec![]
    };

    pb::Frame {
        timestep: f.timestep,
        state: f.state.clone(),
        action: f.action.clone(),
        reward: f.reward.unwrap_or(0.0),
        is_terminal: f.is_terminal,
        images,
    }
}

fn episode_to_pb(ep: &kinodb_core::Episode, include_images: bool) -> pb::Episode {
    pb::Episode {
        meta: Some(meta_to_pb(&ep.meta)),
        frames: ep
            .frames
            .iter()
            .map(|f| frame_to_pb(f, include_images))
            .collect(),
    }
}

// ── gRPC trait impl ─────────────────────────────────────────

#[tonic::async_trait]
impl KinoService for KinoServer {
    async fn get_episode(
        &self,
        request: Request<pb::EpisodeRequest>,
    ) -> Result<Response<pb::EpisodeResponse>, Status> {
        let pos = request.into_inner().position as usize;
        let ep = self.backend.read_episode(pos)?;
        Ok(Response::new(pb::EpisodeResponse {
            episode: Some(episode_to_pb(&ep, true)),
        }))
    }

    async fn get_meta(
        &self,
        request: Request<pb::MetaRequest>,
    ) -> Result<Response<pb::MetaResponse>, Status> {
        let pos = request.into_inner().position as usize;
        let meta = self.backend.read_meta(pos)?;
        Ok(Response::new(pb::MetaResponse {
            meta: Some(meta_to_pb(&meta)),
        }))
    }

    async fn get_batch(
        &self,
        request: Request<pb::BatchRequest>,
    ) -> Result<Response<pb::BatchResponse>, Status> {
        let req = request.into_inner();
        let batch_size = req.batch_size as usize;
        let include_images = req.include_images;

        if batch_size == 0 {
            return Err(Status::invalid_argument("batch_size must be > 0"));
        }

        let mut episodes = Vec::with_capacity(batch_size);

        match req.mode.as_str() {
            "sequential" => {
                let start = req.offset as usize;
                let n = self.backend.num_episodes();
                for i in 0..batch_size {
                    let pos = (start + i) % n;
                    let ep = self.backend.read_episode(pos)?;
                    episodes.push(episode_to_pb(&ep, include_images));
                }
            }
            "random" | "weighted" | "" => {
                for _ in 0..batch_size {
                    let ep = self.backend.sample()?;
                    episodes.push(episode_to_pb(&ep, include_images));
                }
            }
            other => {
                return Err(Status::invalid_argument(format!(
                    "unknown mode '{}'. Use: sequential, random, weighted",
                    other
                )));
            }
        }

        Ok(Response::new(pb::BatchResponse { episodes }))
    }

    async fn query(
        &self,
        request: Request<pb::QueryRequest>,
    ) -> Result<Response<pb::QueryResponse>, Status> {
        let req = request.into_inner();

        let query = kql::parse(&req.kql)
            .map_err(|e| Status::invalid_argument(format!("KQL parse error: {}", e)))?;

        let mut positions = Vec::new();
        let mut metas = Vec::new();

        let n = self.backend.num_episodes();
        let limit = if req.limit > 0 { req.limit as usize } else { n };

        for i in 0..n {
            if positions.len() >= limit {
                break;
            }
            let meta = self.backend.read_meta(i)?;
            if kql::evaluate(&query, &meta) {
                positions.push(i as u64);
                metas.push(meta_to_pb(&meta));
            }
        }

        let total_matches = positions.len() as u64;

        Ok(Response::new(pb::QueryResponse {
            positions,
            metas,
            total_matches,
        }))
    }

    async fn server_info(
        &self,
        _request: Request<pb::InfoRequest>,
    ) -> Result<Response<pb::InfoResponse>, Status> {
        Ok(Response::new(pb::InfoResponse {
            num_episodes: self.backend.num_episodes() as u64,
            num_frames: self.backend.num_frames(),
            num_sources: self.backend.num_sources(),
            version: "0.1.0".to_string(),
            sources: self.backend.source_names(),
        }))
    }
}
