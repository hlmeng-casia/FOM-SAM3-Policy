/**
 * Paper-backed content and external-resource configuration.
 * Empty link strings are disabled automatically by the UI.
 */
window.PROJECT_DATA = {
  shortName: "FOM-SAM3",
  title: "Towards Fine-Grained Object Manipulation: SAM3-Guided Visuomotor Policy with Persistent Memory Learning and Focused Visual Conditioning",
  label: "FOM-SAM3 · Project Page",
  authors: ["Anonymous Authors"],
  institutions: ["Anonymous Institution"],
  summary:
    "FO Memory gives frozen SAM3 a persistent fine-grained identity, while Focused Spatial-Appearance Encoding turns the selected target and its task context into policy-ready conditions for DP or ACT.",
  abstract: [
    "Imitation learning of visuomotor policies endows the robot with dexterous manipulation skills. However, even in a familiar scene where a robot repeats its tasks, distinguishing a target fine-grained object (FO) from similar objects in the same coarse category remains an unresolved issue. Besides, current visuomotor policies encode the scene image as the visual condition, which might be prone to visual distractors. To address these problems, we first propose FO Memory-enhanced Segment Anything Model 3 (FOM-SAM3). With SAM3 fully frozen, it converts a coarse concept prompt into reusable FO Memory Tokens from a few multi-view registration images. At deployment, the tokens directly prompt frozen SAM3 without registration images or model adaptation. Second, we introduce Focused Spatial-Appearance Encoding (FSAE), which focuses the action policy on ordered target and manipulation-context regions and models their local appearance and relative geometry. FSAE produces a compact condition vector for DP or object condition tokens for ACT without changing either action objective. Experiments on 30 FOs from 4 coarse categories show that FO Memory improves target segmentation, ranks the queried FO above visually similar candidates, suppresses Similar-FO confusion, and rejects absent identities more reliably. Real-robot experiments report completed and failed rollouts and evaluate within-type new-FO reuse with frozen policies.",
  ],
  abstractHighlights: [
    "fine-grained object (FO)",
    "FO Memory-enhanced Segment Anything Model 3 (FOM-SAM3)",
    "reusable FO Memory Tokens",
    "Focused Spatial-Appearance Encoding (FSAE)",
    "30 FOs from 4 coarse categories",
    "Real-robot experiments",
  ],
  links: {
    paper: "main.pdf",
    arxiv: "",
    code: "",
    video: "#demonstrations-title",
    dataset: "",
    demo: "",
  },
  videoPresentation: {
    youtubeId: "1QNsdr-Qx_I",
  },
  demoVideos: [
    { src: "assets/videos/7120667f7ee628d3572718b42e4181d4-web.mp4", poster: "assets/videos/7120667f7ee628d3572718b42e4181d4_thumb.jpg", title: "Robot demonstration 01" },
    { src: "assets/videos/761b332e22c9c3354ad8dd4cb17496bd-web.mp4", poster: "assets/videos/761b332e22c9c3354ad8dd4cb17496bd_thumb.jpg", title: "Robot demonstration 02" },
    { src: "assets/videos/99c1d6ad937239951173e9a425089df7-web.mp4", poster: "assets/videos/99c1d6ad937239951173e9a425089df7_thumb.jpg", title: "Robot demonstration 03" },
    { src: "assets/videos/bb7614bb9cb0407790399c410935142e-web.mp4", poster: "assets/videos/bb7614bb9cb0407790399c410935142e_thumb.jpg", title: "Robot demonstration 04" },
    { src: "assets/videos/cc3d9e164d8125a42e45f3658c80bbf6-web.mp4", poster: "assets/videos/cc3d9e164d8125a42e45f3658c80bbf6_thumb.jpg", title: "Robot demonstration 05" },
    { src: "assets/videos/dff5ab82737ffcf5eb01ed6258cc3b16-web.mp4", poster: "assets/videos/dff5ab82737ffcf5eb01ed6258cc3b16_thumb.jpg", title: "Robot demonstration 06" },
    { src: "assets/videos/e03611c5f4cf1207a428a79dc9ac6bb6-web.mp4", poster: "assets/videos/e03611c5f4cf1207a428a79dc9ac6bb6_thumb.jpg", title: "Robot demonstration 07" },
  ],
  generalizationDemos: [
    { group: "New-FO reuse", title: "Pick & Place", videoIndex: 0 },
    { group: "New-FO reuse", title: "Push", videoIndex: 1 },
    { group: "New-FO reuse", title: "Assemble", videoIndex: 2 },
    { group: "Identity & instance variation", title: "Similar-FO selection", videoIndex: 3 },
    { group: "Identity & instance variation", title: "Novel instance", videoIndex: 4 },
    { group: "Identity & instance variation", title: "Large appearance shift", videoIndex: 5 },
    { group: "Background generalization", title: "Seen background", videoIndex: 6 },
    { group: "Background generalization", title: "Background shift", videoIndex: 0 },
    { group: "Background generalization", title: "Clutter", videoIndex: 1 },
  ],
  taskPrompts: {
    "Pick & Place": "red CocaCola soda can",
    Push: "white pill box",
    Assemble: "cup lid with panda handle",
  },
  robotResults: [
    { method: "RGB-DP", id: ["8/10", "7/10", "5/10"], distractors: ["6/10", "3/10", "5/10"], similar: ["1/10", "3/10", "0/10"] },
    { method: "S²-Diff.", id: ["10/10", "7/10", "4/10"], distractors: ["8/10", "6/10", "5/10"], similar: ["4/10", "6/10", "5/10"] },
    { method: "Ours-DP", id: ["10/10", "9/10", "8/10"], distractors: ["10/10", "8/10", "10/10"], similar: ["10/10", "10/10", "10/10"], ours: true },
    { method: "RGB-ACT", id: ["10/10", "8/10", "6/10"], distractors: ["4/10", "5/10", "3/10"], similar: ["0/10", "5/10", "1/10"] },
    { method: "Ours-ACT", id: ["10/10", "9/10", "9/10"], distractors: ["8/10", "10/10", "9/10"], similar: ["8/10", "9/10", "7/10"], ours: true },
  ],
  citation: `@article{anonymous2026fomsam3,
  title  = {Towards Fine-Grained Object Manipulation: SAM3-Guided
            Visuomotor Policy with Persistent Memory Learning and Focused Visual Conditioning},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Project page}
}`,
};
