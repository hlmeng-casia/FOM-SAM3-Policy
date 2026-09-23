/**
 * Paper-backed content and external-resource configuration.
 * Empty link strings are disabled automatically by the UI.
 */
window.PROJECT_DATA = {
  title: "Towards Fine-Grained Object Manipulation: SAM3-Guided Visuomotor Policy with Persistent Memory Learning and Focused Visual Conditioning",
  abstract: [
    "Fine-grained object (FO) manipulation requires robots to distinguish a specified FO from visually similar objects and execute actions reliably despite scene distractors. However, scene-level visual conditioning lacks explicit object selection, while category-level guidance cannot reliably distinguish FOs within the same category. We present a SAM3-guided visuomotor framework that addresses these challenges through persistent object memory and focused visual conditioning. First, we introduce FO Memory-driven SAM3 (FOM-SAM3), which learns reusable FO memory tokens from limited multi-view registration images while keeping SAM3 fully frozen. Through one-vs-rest learning, these tokens encode persistent memories for localizing target FOs and rejecting similar alternatives, which can be stored in a memory bank. Second, we propose Focused Spatial-Appearance Encoding (FSAE), which combines in-FO local appearance features with explicit bounding-box coordinates to condition action policies including Diffusion Policy (DP) and Action Chunking with Transformers (ACT). The effectiveness of the proposed FOM-SAM3 was validated on the FO-30 dataset comprising 30 physical objects across four coarse categories. Across three real-robot FO manipulation tasks, our FOM-SAM3-guided policies demonstrated the robustness against distractors, discrimination ability among similar FOs, and extendibility to new FOs.",
  ],
  abstractHighlights: [
    "fine-grained object (FO)",
    "FO Memory-driven SAM3 (FOM-SAM3)",
    "reusable FO memory tokens",
    "Focused Spatial-Appearance Encoding (FSAE)",
    "FO-30 dataset",
    "three real-robot FO manipulation tasks",
  ],
  links: {
    video: "#main-video",
  },
  demoVideos: [
    { src: "assets/videos/1.mp4", title: "Collect Can · Continuous long-horizon rollout" },
    { src: "assets/videos/2.mp4", title: "Push Box · Continuous long-horizon rollout" },
    { src: "assets/videos/3.mp4", title: "Lid Cup · Continuous long-horizon rollout" },
    { src: "assets/videos/4.mp4", title: "Collect Can · Cluttered scene" },
    { src: "assets/videos/5.mp4", title: "Collect Can · Background change" },
    { src: "assets/videos/fsae_1.mp4", title: "FSAE Visualization 1" },
    { src: "assets/videos/fsae_2.mp4", title: "FSAE Visualization 2" },
    { src: "assets/videos/fsae_3.mp4", title: "FSAE Visualization 3" },
    { src: "assets/videos/fsae_4.mp4", title: "FSAE Visualization 4" },
  ],
  taskPrompts: {
    "Collect Can": "red CocaCola soda can",
    "Push Box": "white pill box",
    "Lid Cup": "cup lid with panda handle",
  },
  robotResults: [
    { method: "RGB-DP", id: ["8/10", "7/10", "5/10"], distractors: ["6/10", "3/10", "5/10"], similar: ["1/10", "3/10", "0/10"] },
    { method: "S²-Diff.", id: ["10/10", "7/10", "4/10"], distractors: ["8/10", "6/10", "5/10"], similar: ["4/10", "6/10", "5/10"] },
    { method: "Ours-DP", id: ["10/10", "9/10", "8/10"], distractors: ["10/10", "8/10", "10/10"], similar: ["10/10", "10/10", "10/10"], ours: true },
    { method: "RGB-ACT", id: ["10/10", "8/10", "6/10"], distractors: ["4/10", "5/10", "3/10"], similar: ["0/10", "5/10", "1/10"] },
    { method: "Ours-ACT", id: ["10/10", "9/10", "9/10"], distractors: ["8/10", "10/10", "9/10"], similar: ["8/10", "9/10", "7/10"], ours: true },
  ],
};
