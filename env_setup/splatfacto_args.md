ns-train splatfacto --help
usage: /home/computer0/anaconda3/envs/fly_gsplat/bin/ns-train splatfacto [-h] [SPLATFACTO OPTIONS] [{nerfstudio-data,minimal-parser,arkit-data,blender-data,instant-ngp-data,nuscenes-data,dnerf-data,phototourism-data,dycheck-data,scannet-data,sdfstudio-data,nerfosr-data,sitcoms3d-data,scannetpp-data,colmap}]

Gaussian Splatting model

╭─ options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help                                                                                                           │
│       show this help message and exit                                                                                │
│ --output-dir PATH                                                                                                    │
│       relative or absolute output directory to save all checkpoints and logging (default: outputs)                   │
│ --method-name {None}|STR                                                                                             │
│       Method name. Required to set in python or via cli (default: splatfacto)                                        │
│ --experiment-name {None}|STR                                                                                         │
│       Experiment name. If None, will automatically be set to dataset name (default: None)                            │
│ --project-name {None}|STR                                                                                            │
│       Project name. (default: nerfstudio-project)                                                                    │
│ --timestamp STR                                                                                                      │
│       Experiment timestamp. (default: '{timestamp}')                                                                 │
│ --vis {viewer,wandb,tensorboard,comet,viewer+wandb,viewer+tensorboard,viewer+comet,viewer_legacy}                    │
│       Which visualizer to use. (default: viewer)                                                                     │
│ --data {None}|PATH                                                                                                   │
│       Alias for --pipeline.datamanager.data (default: None)                                                          │
│ --prompt {None}|STR                                                                                                  │
│       Alias for --pipeline.model.prompt (default: None)                                                              │
│ --relative-model-dir PATH                                                                                            │
│       Relative path to save all checkpoints. (default: nerfstudio_models)                                            │
│ --load-scheduler {True,False}                                                                                        │
│       Whether to load the scheduler state_dict to resume training, if it exists. (default: True)                     │
│ --steps-per-save INT                                                                                                 │
│       Number of steps between saves. (default: 2000)                                                                 │
│ --steps-per-eval-batch INT                                                                                           │
│       Number of steps between randomly sampled batches of rays. (default: 0)                                         │
│ --steps-per-eval-image INT                                                                                           │
│       Number of steps between single eval images. (default: 100)                                                     │
│ --steps-per-eval-all-images INT                                                                                      │
│       Number of steps between eval all images. (default: 1000)                                                       │
│ --max-num-iterations INT                                                                                             │
│       Maximum number of iterations to run. (default: 30000)                                                          │
│ --mixed-precision {True,False}                                                                                       │
│       Whether or not to use mixed precision for training. (default: False)                                           │
│ --use-grad-scaler {True,False}                                                                                       │
│       Use gradient scaler even if the automatic mixed precision is disabled. (default: False)                        │
│ --save-only-latest-checkpoint {True,False}                                                                           │
│       Whether to only save the latest checkpoint or all checkpoints. (default: True)                                 │
│ --load-dir {None}|PATH                                                                                               │
│       Optionally specify a pre-trained model directory to load from. (default: None)                                 │
│ --load-step {None}|INT                                                                                               │
│       Optionally specify model step to load from; if none, will find most recent model in load_dir. (default: None)  │
│ --load-config {None}|PATH                                                                                            │
│       Path to config YAML file. (default: None)                                                                      │
│ --load-checkpoint {None}|PATH                                                                                        │
│       Path to checkpoint file. (default: None)                                                                       │
│ --log-gradients {True,False}                                                                                         │
│       Optionally log gradients during training (default: False)                                                      │
│ --gradient-accumulation-steps [STR INT [STR INT ...]]                                                                │
│       Number of steps to accumulate gradients over. Contains a mapping of {param_group:num} (default: )              │
│ --start-paused {True,False}                                                                                          │
│       Whether to start the training in a paused state. (default: False)                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ machine options ────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Machine configuration                                                                                                │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --machine.seed INT                                                                                                   │
│       random seed initialization (default: 42)                                                                       │
│ --machine.num-devices INT                                                                                            │
│       total number of devices (e.g., gpus) available for train/eval (default: 1)                                     │
│ --machine.num-machines INT                                                                                           │
│       total number of distributed machines available (for DDP) (default: 1)                                          │
│ --machine.machine-rank INT                                                                                           │
│       current machine's rank (for DDP) (default: 0)                                                                  │
│ --machine.dist-url STR                                                                                               │
│       distributed connection point (for DDP) (default: auto)                                                         │
│ --machine.device-type {cpu,cuda,mps}                                                                                 │
│       device type to use for training (default: cuda)                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ logging options ────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Logging configuration                                                                                                │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --logging.relative-log-dir PATH                                                                                      │
│       relative path to save all logged events (default: .)                                                           │
│ --logging.steps-per-log INT                                                                                          │
│       number of steps between logging stats (default: 10)                                                            │
│ --logging.max-buffer-size INT                                                                                        │
│       maximum history size to keep for computing running averages of stats.                                          │
│       e.g. if 20, averages will be computed over past 20 occurrences. (default: 20)                                  │
│ --logging.profiler {none,basic,pytorch}                                                                              │
│       how to profile the code;                                                                                       │
│       "basic" - prints speed of all decorated functions at the end of a program.                                     │
│       "pytorch" - same as basic, but it also traces few training steps. (default: basic)                             │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ logging.local-writer options ───────────────────────────────────────────────────────────────────────────────────────╮
│ if provided, will print stats locally. if None, will disable printing                                                │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --logging.local-writer.enable {True,False}                                                                           │
│       if True enables local logging, else disables (default: True)                                                   │
│ --logging.local-writer.stats-to-track [{ITER_TRAIN_TIME,TOTAL_TRAIN_TIME,ETA,TRAIN_RAYS_PER_SEC,TEST_RAYS_PER_SEC,   │
│ VIS_RAYS_PER_SEC,CURR_TEST_PSNR} [...]]                                                                              │
│       specifies which stats will be logged/printed to terminal (default: ITER_TRAIN_TIME TRAIN_RAYS_PER_SEC          │
│       CURR_TEST_PSNR VIS_RAYS_PER_SEC TEST_RAYS_PER_SEC ETA)                                                         │
│ --logging.local-writer.max-log-size INT                                                                              │
│       maximum number of rows to print before wrapping. if 0, will print everything. (default: 10)                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ viewer options ─────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Viewer configuration                                                                                                 │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --viewer.relative-log-filename STR                                                                                   │
│       Filename to use for the log file. (default: viewer_log_filename.txt)                                           │
│ --viewer.websocket-port {None}|INT                                                                                   │
│       The websocket port to connect to. If None, find an available port. (default: None)                             │
│ --viewer.websocket-port-default INT                                                                                  │
│       The default websocket port to connect to if websocket_port is not specified (default: 7007)                    │
│ --viewer.websocket-host STR                                                                                          │
│       The host address to bind the websocket server to. (default: 0.0.0.0)                                           │
│ --viewer.num-rays-per-chunk INT                                                                                      │
│       number of rays per chunk to render with viewer (default: 32768)                                                │
│ --viewer.max-num-display-images INT                                                                                  │
│       Maximum number of training images to display in the viewer, to avoid lag. This does not change which images    │
│       are                                                                                                            │
│       actually used in training/evaluation. If -1, display all. (default: 512)                                       │
│ --viewer.quit-on-train-completion {True,False}                                                                       │
│       Whether to kill the training job when it has completed. Note this will stop rendering in the viewer. (default: │
│       False)                                                                                                         │
│ --viewer.image-format {jpeg,png}                                                                                     │
│       Image format viewer should use; jpeg is lossy compression, while png is lossless. (default: jpeg)              │
│ --viewer.jpeg-quality INT                                                                                            │
│       Quality tradeoff to use for jpeg compression. (default: 75)                                                    │
│ --viewer.make-share-url {True,False}                                                                                 │
│       Viewer beta feature: print a shareable URL. This flag is ignored in the legacy version of the viewer.          │
│       (default: False)                                                                                               │
│ --viewer.camera-frustum-scale FLOAT                                                                                  │
│       Scale for the camera frustums in the viewer. (default: 0.1)                                                    │
│ --viewer.default-composite-depth {True,False}                                                                        │
│       The default value for compositing depth. Turn off if you want to see the camera frustums without occlusions.   │
│       (default: True)                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ pipeline.datamanager options ───────────────────────────────────────────────────────────────────────────────────────╮
│ specifies the datamanager config                                                                                     │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --pipeline.datamanager.data {None}|PATH                                                                              │
│       Source of data, may not be used by all models. (default: None)                                                 │
│ --pipeline.datamanager.masks-on-gpu {True,False}                                                                     │
│       Process masks on GPU for speed at the expense of memory, if True. (default: False)                             │
│ --pipeline.datamanager.images-on-gpu {True,False}                                                                    │
│       Process images on GPU for speed at the expense of memory, if True. (default: False)                            │
│ --pipeline.datamanager.camera-res-scale-factor FLOAT                                                                 │
│       The scale factor for scaling spatial data such as images, mask, semantics                                      │
│       along with relevant information about camera intrinsics (default: 1.0)                                         │
│ --pipeline.datamanager.eval-num-images-to-sample-from INT                                                            │
│       Number of images to sample during eval iteration. (default: -1)                                                │
│ --pipeline.datamanager.eval-num-times-to-repeat-images INT                                                           │
│       When not evaluating on all images, number of iterations before picking                                         │
│       new images. If -1, never pick new images. (default: -1)                                                        │
│ --pipeline.datamanager.eval-image-indices {None}|{[INT [INT ...]]}                                                   │
│       Specifies the image indices to use during eval; if None, uses all. (default: 0)                                │
│ --pipeline.datamanager.cache-images {cpu,gpu}                                                                        │
│       Whether to cache images in memory. If "cpu", caches on cpu. If "gpu", caches on device. (default: gpu)         │
│ --pipeline.datamanager.cache-images-type {uint8,float32}                                                             │
│       The image type returned from manager, caching images in uint8 saves memory (default: uint8)                    │
│ --pipeline.datamanager.max-thread-workers {None}|INT                                                                 │
│       The maximum number of threads to use for caching images. If None, uses all available threads. (default: None)  │
│ --pipeline.datamanager.train-cameras-sampling-strategy {random,fps}                                                  │
│       Specifies which sampling strategy is used to generate train cameras, 'random' means sampling                   │
│       uniformly random without replacement, 'fps' means farthest point sampling which is helpful to reduce the       │
│       artifacts                                                                                                      │
│       due to oversampling subsets of cameras that are very close to each other. (default: random)                    │
│ --pipeline.datamanager.train-cameras-sampling-seed INT                                                               │
│       Random seed for sampling train cameras. Fixing seed may help reduce variance of trained models across          │
│       different runs. (default: 42)                                                                                  │
│ --pipeline.datamanager.fps-reset-every INT                                                                           │
│       The number of iterations before one resets fps sampler repeatly, which is essentially drawing fps_reset_every  │
│       samples from the pool of all training cameras without replacement before a new round of sampling starts.       │
│       (default: 100)                                                                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ pipeline.model options ─────────────────────────────────────────────────────────────────────────────────────────────╮
│ specifies the model config                                                                                           │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --pipeline.model.enable-collider {True,False}                                                                        │
│       Whether to create a scene collider to filter rays. (default: True)                                             │
│ --pipeline.model.collider-params {None}|{[STR FLOAT [STR FLOAT ...]]}                                                │
│       parameters to instantiate scene collider with (default: near_plane 2.0 far_plane 6.0)                          │
│ --pipeline.model.eval-num-rays-per-chunk INT                                                                         │
│       specifies number of rays per chunk during eval (default: 4096)                                                 │
│ --pipeline.model.prompt {None}|STR                                                                                   │
│       A prompt to be used in text to NeRF models (default: None)                                                     │
│ --pipeline.model.warmup-length INT                                                                                   │
│       period of steps where refinement is turned off (default: 500)                                                  │
│ --pipeline.model.refine-every INT                                                                                    │
│       period of steps where gaussians are culled and densified (default: 100)                                        │
│ --pipeline.model.resolution-schedule INT                                                                             │
│       training starts at 1/d resolution, every n steps this is doubled (default: 3000)                               │
│ --pipeline.model.background-color {random,black,white}                                                               │
│       Whether to randomize the background color. (default: random)                                                   │
│ --pipeline.model.num-downscales INT                                                                                  │
│       at the beginning, resolution is 1/2^d, where d is this number (default: 2)                                     │
│ --pipeline.model.cull-alpha-thresh FLOAT                                                                             │
│       threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality.   │
│       (default: 0.1)                                                                                                 │
│ --pipeline.model.cull-scale-thresh FLOAT                                                                             │
│       threshold of scale for culling huge gaussians (default: 0.5)                                                   │
│ --pipeline.model.reset-alpha-every INT                                                                               │
│       Every this many refinement steps, reset the alpha (default: 30)                                                │
│ --pipeline.model.densify-grad-thresh FLOAT                                                                           │
│       threshold of positional gradient norm for densifying gaussians (default: 0.0008)                               │
│ --pipeline.model.use-absgrad {True,False}                                                                            │
│       Whether to use absgrad to densify gaussians, if False, will use grad rather than absgrad (default: True)       │
│ --pipeline.model.densify-size-thresh FLOAT                                                                           │
│       below this size, gaussians are *duplicated*, otherwise split (default: 0.01)                                   │
│ --pipeline.model.n-split-samples INT                                                                                 │
│       number of samples to split gaussians into (default: 2)                                                         │
│ --pipeline.model.sh-degree-interval INT                                                                              │
│       every n intervals turn on another sh degree (default: 1000)                                                    │
│ --pipeline.model.cull-screen-size FLOAT                                                                              │
│       if a gaussian is more than this percent of screen space, cull it (default: 0.15)                               │
│ --pipeline.model.split-screen-size FLOAT                                                                             │
│       if a gaussian is more than this percent of screen space, split it (default: 0.05)                              │
│ --pipeline.model.stop-screen-size-at INT                                                                             │
│       stop culling/splitting at this step WRT screen size of gaussians (default: 4000)                               │
│ --pipeline.model.random-init {True,False}                                                                            │
│       whether to initialize the positions uniformly randomly (not SFM points) (default: False)                       │
│ --pipeline.model.num-random INT                                                                                      │
│       Number of gaussians to initialize if random init is used (default: 50000)                                      │
│ --pipeline.model.random-scale FLOAT                                                                                  │
│       Size of the cube to initialize random gaussians within (default: 10.0)                                         │
│ --pipeline.model.ssim-lambda FLOAT                                                                                   │
│       weight of ssim loss (default: 0.2)                                                                             │
│ --pipeline.model.stop-split-at INT                                                                                   │
│       stop splitting at this step (default: 15000)                                                                   │
│ --pipeline.model.sh-degree INT                                                                                       │
│       maximum degree of spherical harmonics to use (default: 3)                                                      │
│ --pipeline.model.use-scale-regularization {True,False}                                                               │
│       If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used  │
│       for reducing huge spikey gaussians. (default: False)                                                           │
│ --pipeline.model.max-gauss-ratio FLOAT                                                                               │
│       threshold of ratio of gaussian max to min scale before applying regularization                                 │
│       loss from the PhysGaussian paper (default: 10.0)                                                               │
│ --pipeline.model.output-depth-during-training {True,False}                                                           │
│       If True, output depth during training. Otherwise, only output depth during evaluation. (default: False)        │
│ --pipeline.model.rasterize-mode {classic,antialiased}                                                                │
│       Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel.    │
│       This                                                                                                           │
│       approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured,     │
│       which                                                                                                          │
│       results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation  │
│       factors                                                                                                        │
│       and apply them to the opacities of gaussians to preserve the total integrated density of splats.               │
│                                                                                                                      │
│                                                                                                                      │
│       However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web       │
│       viewers that                                                                                                   │
│       were implemented for classic mode can not render antialiased mode PLY properly without modifications.          │
│       (default: classic)                                                                                             │
│ --pipeline.model.use-bilateral-grid {True,False}                                                                     │
│       If True, use bilateral grid to handle the ISP changes in the image space. This technique was introduced in the │
│       paper 'Bilateral Guided Radiance Field Processing' (https://bilarfpro.github.io/). (default: False)            │
│ --pipeline.model.grid-shape INT INT INT                                                                              │
│       Shape of the bilateral grid (X, Y, W) (default: 16 16 8)                                                       │
│ --pipeline.model.color-corrected-metrics {True,False}                                                                │
│       If True, apply color correction to the rendered images before computing the metrics. (default: False)          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ pipeline.model.loss-coefficients options ───────────────────────────────────────────────────────────────────────────╮
│ parameters to instantiate density field with                                                                         │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --pipeline.model.loss-coefficients.rgb-loss-coarse FLOAT                                                             │
│       (default: 1.0)                                                                                                 │
│ --pipeline.model.loss-coefficients.rgb-loss-fine FLOAT                                                               │
│       (default: 1.0)                                                                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ pipeline.model.camera-optimizer options ────────────────────────────────────────────────────────────────────────────╮
│ Config of the camera optimizer to use                                                                                │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --pipeline.model.camera-optimizer.mode {off,SO3xR3,SE3}                                                              │
│       Pose optimization strategy to use. If enabled, we recommend SO3xR3. (default: off)                             │
│ --pipeline.model.camera-optimizer.trans-l2-penalty FLOAT                                                             │
│       L2 penalty on translation parameters. (default: 0.01)                                                          │
│ --pipeline.model.camera-optimizer.rot-l2-penalty FLOAT                                                               │
│       L2 penalty on rotation parameters. (default: 0.001)                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.means.optimizer options ─────────────────────────────────────────────────────────────────────────────────╮
│ Basic optimizer config with Adam                                                                                     │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --optimizers.means.optimizer.lr FLOAT                                                                                │
│       The learning rate to use. (default: 0.00016)                                                                   │
│ --optimizers.means.optimizer.eps FLOAT                                                                               │
│       The epsilon value to use. (default: 1e-15)                                                                     │
│ --optimizers.means.optimizer.max-norm {None}|FLOAT                                                                   │
│       The max norm to use for gradient clipping. (default: None)                                                     │
│ --optimizers.means.optimizer.weight-decay FLOAT                                                                      │
│       The weight decay to use. (default: 0)                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.means.scheduler options ─────────────────────────────────────────────────────────────────────────────────╮
│ Config for exponential decay scheduler with warmup                                                                   │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --optimizers.means.scheduler.lr-pre-warmup FLOAT                                                                     │
│       Learning rate before warmup. (default: 1e-08)                                                                  │
│ --optimizers.means.scheduler.lr-final {None}|FLOAT                                                                   │
│       Final learning rate. If not provided, it will be set to the optimizers learning rate. (default: 1.6e-06)       │
│ --optimizers.means.scheduler.warmup-steps INT                                                                        │
│       Number of warmup steps. (default: 0)                                                                           │
│ --optimizers.means.scheduler.max-steps INT                                                                           │
│       The maximum number of steps. (default: 30000)                                                                  │
│ --optimizers.means.scheduler.ramp {linear,cosine}                                                                    │
│       The ramp function to use during the warmup. (default: cosine)                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.features-dc options ─────────────────────────────────────────────────────────────────────────────────────╮
│ --optimizers.features-dc.scheduler {None}                                                                            │
│       (default: None)                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.features-dc.optimizer options ───────────────────────────────────────────────────────────────────────────╮
│ Basic optimizer config with Adam                                                                                     │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --optimizers.features-dc.optimizer.lr FLOAT                                                                          │
│       The learning rate to use. (default: 0.0025)                                                                    │
│ --optimizers.features-dc.optimizer.eps FLOAT                                                                         │
│       The epsilon value to use. (default: 1e-15)                                                                     │
│ --optimizers.features-dc.optimizer.max-norm {None}|FLOAT                                                             │
│       The max norm to use for gradient clipping. (default: None)                                                     │
│ --optimizers.features-dc.optimizer.weight-decay FLOAT                                                                │
│       The weight decay to use. (default: 0)                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.features-rest options ───────────────────────────────────────────────────────────────────────────────────╮
│ --optimizers.features-rest.scheduler {None}                                                                          │
│       (default: None)                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.features-rest.optimizer options ─────────────────────────────────────────────────────────────────────────╮
│ Basic optimizer config with Adam                                                                                     │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --optimizers.features-rest.optimizer.lr FLOAT                                                                        │
│       The learning rate to use. (default: 0.000125)                                                                  │
│ --optimizers.features-rest.optimizer.eps FLOAT                                                                       │
│       The epsilon value to use. (default: 1e-15)                                                                     │
│ --optimizers.features-rest.optimizer.max-norm {None}|FLOAT                                                           │
│       The max norm to use for gradient clipping. (default: None)                                                     │
│ --optimizers.features-rest.optimizer.weight-decay FLOAT                                                              │
│       The weight decay to use. (default: 0)                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.opacities options ───────────────────────────────────────────────────────────────────────────────────────╮
│ --optimizers.opacities.scheduler {None}                                                                              │
│       (default: None)                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.opacities.optimizer options ─────────────────────────────────────────────────────────────────────────────╮
│ Basic optimizer config with Adam                                                                                     │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --optimizers.opacities.optimizer.lr FLOAT                                                                            │
│       The learning rate to use. (default: 0.05)                                                                      │
│ --optimizers.opacities.optimizer.eps FLOAT                                                                           │
│       The epsilon value to use. (default: 1e-15)                                                                     │
│ --optimizers.opacities.optimizer.max-norm {None}|FLOAT                                                               │
│       The max norm to use for gradient clipping. (default: None)                                                     │
│ --optimizers.opacities.optimizer.weight-decay FLOAT                                                                  │
│       The weight decay to use. (default: 0)                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.scales options ──────────────────────────────────────────────────────────────────────────────────────────╮
│ --optimizers.scales.scheduler {None}                                                                                 │
│       (default: None)                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.scales.optimizer options ────────────────────────────────────────────────────────────────────────────────╮
│ Basic optimizer config with Adam                                                                                     │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --optimizers.scales.optimizer.lr FLOAT                                                                               │
│       The learning rate to use. (default: 0.005)                                                                     │
│ --optimizers.scales.optimizer.eps FLOAT                                                                              │
│       The epsilon value to use. (default: 1e-15)                                                                     │
│ --optimizers.scales.optimizer.max-norm {None}|FLOAT                                                                  │
│       The max norm to use for gradient clipping. (default: None)                                                     │
│ --optimizers.scales.optimizer.weight-decay FLOAT                                                                     │
│       The weight decay to use. (default: 0)                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.quats options ───────────────────────────────────────────────────────────────────────────────────────────╮
│ --optimizers.quats.scheduler {None}                                                                                  │
│       (default: None)                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.quats.optimizer options ─────────────────────────────────────────────────────────────────────────────────╮
│ Basic optimizer config with Adam                                                                                     │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --optimizers.quats.optimizer.lr FLOAT                                                                                │
│       The learning rate to use. (default: 0.001)                                                                     │
│ --optimizers.quats.optimizer.eps FLOAT                                                                               │
│       The epsilon value to use. (default: 1e-15)                                                                     │
│ --optimizers.quats.optimizer.max-norm {None}|FLOAT                                                                   │
│       The max norm to use for gradient clipping. (default: None)                                                     │
│ --optimizers.quats.optimizer.weight-decay FLOAT                                                                      │
│       The weight decay to use. (default: 0)                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.camera-opt.optimizer options ────────────────────────────────────────────────────────────────────────────╮
│ Basic optimizer config with Adam                                                                                     │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --optimizers.camera-opt.optimizer.lr FLOAT                                                                           │
│       The learning rate to use. (default: 0.0001)                                                                    │
│ --optimizers.camera-opt.optimizer.eps FLOAT                                                                          │
│       The epsilon value to use. (default: 1e-15)                                                                     │
│ --optimizers.camera-opt.optimizer.max-norm {None}|FLOAT                                                              │
│       The max norm to use for gradient clipping. (default: None)                                                     │
│ --optimizers.camera-opt.optimizer.weight-decay FLOAT                                                                 │
│       The weight decay to use. (default: 0)                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.camera-opt.scheduler options ────────────────────────────────────────────────────────────────────────────╮
│ Config for exponential decay scheduler with warmup                                                                   │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --optimizers.camera-opt.scheduler.lr-pre-warmup FLOAT                                                                │
│       Learning rate before warmup. (default: 0)                                                                      │
│ --optimizers.camera-opt.scheduler.lr-final {None}|FLOAT                                                              │
│       Final learning rate. If not provided, it will be set to the optimizers learning rate. (default: 5e-07)         │
│ --optimizers.camera-opt.scheduler.warmup-steps INT                                                                   │
│       Number of warmup steps. (default: 1000)                                                                        │
│ --optimizers.camera-opt.scheduler.max-steps INT                                                                      │
│       The maximum number of steps. (default: 30000)                                                                  │
│ --optimizers.camera-opt.scheduler.ramp {linear,cosine}                                                               │
│       The ramp function to use during the warmup. (default: cosine)                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.bilateral-grid.optimizer options ────────────────────────────────────────────────────────────────────────╮
│ Basic optimizer config with Adam                                                                                     │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --optimizers.bilateral-grid.optimizer.lr FLOAT                                                                       │
│       The learning rate to use. (default: 0.002)                                                                     │
│ --optimizers.bilateral-grid.optimizer.eps FLOAT                                                                      │
│       The epsilon value to use. (default: 1e-15)                                                                     │
│ --optimizers.bilateral-grid.optimizer.max-norm {None}|FLOAT                                                          │
│       The max norm to use for gradient clipping. (default: None)                                                     │
│ --optimizers.bilateral-grid.optimizer.weight-decay FLOAT                                                             │
│       The weight decay to use. (default: 0)                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ optimizers.bilateral-grid.scheduler options ────────────────────────────────────────────────────────────────────────╮
│ Config for exponential decay scheduler with warmup                                                                   │
│ ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ --optimizers.bilateral-grid.scheduler.lr-pre-warmup FLOAT                                                            │
│       Learning rate before warmup. (default: 0)                                                                      │
│ --optimizers.bilateral-grid.scheduler.lr-final {None}|FLOAT                                                          │
│       Final learning rate. If not provided, it will be set to the optimizers learning rate. (default: 0.0001)        │
│ --optimizers.bilateral-grid.scheduler.warmup-steps INT                                                               │
│       Number of warmup steps. (default: 1000)                                                                        │
│ --optimizers.bilateral-grid.scheduler.max-steps INT                                                                  │
│       The maximum number of steps. (default: 30000)                                                                  │
│ --optimizers.bilateral-grid.scheduler.ramp {linear,cosine}                                                           │
│       The ramp function to use during the warmup. (default: cosine)                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ subcommands ────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ (default: nerfstudio-data)                                                                                           │
│   • nerfstudio-data                                                                                                  │
│   • minimal-parser                                                                                                   │
│   • arkit-data                                                                                                       │
│   • blender-data                                                                                                     │
│   • instant-ngp-data                                                                                                 │
│   • nuscenes-data                                                                                                    │
│   • dnerf-data                                                                                                       │
│   • phototourism-data                                                                                                │
│   • dycheck-data                                                                                                     │
│   • scannet-data                                                                                                     │
│   • sdfstudio-data                                                                                                   │
│   • nerfosr-data                                                                                                     │
│   • sitcoms3d-data                                                                                                   │
│   • scannetpp-data                                                                                                   │
│   • colmap                                                                                                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯