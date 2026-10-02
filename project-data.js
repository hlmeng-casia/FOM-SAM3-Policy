/**
 * Paper-backed content and external-resource configuration.
 * Empty link strings are disabled automatically by the UI.
 */
window.PROJECT_DATA = {
  title: "Towards Fine-Grained Object Manipulation: SAM3-Guided Visuomotor Policy with Persistent Memory Learning and Focused Visual Conditioning",
  abstract: [
    "Fine-grained object (FO) manipulation requires robots to distinguish a specified FO from visually similar objects and execute actions reliably despite scene distractors. However, scene-level visual conditioning lacks explicit object selection, while category-level guidance cannot reliably distinguish FOs within the same category. We present a SAM3-guided visuomotor policy framework that addresses these challenges through persistent object memory and focused visual conditioning. First, we introduce FO Memory-driven SAM3 (FOM-SAM3), which learns reusable FO memory tokens from limited multi-view registration images, while keeping SAM3 fully frozen. Through one-vs-rest learning, these tokens encode persistent memories for localizing target FOs and rejecting similar alternatives, which can be stored in a memory bank for reuse. Second, we propose Focused Spatial-Appearance Encoding (FSAE), which combines in-FO appearance features with explicit bounding boxes to condition action policies, including Diffusion Policy (DP) and Action Chunking with Transformers (ACT). The effectiveness of the proposed FOM-SAM3 is validated on the FO-30 dataset comprising 30 physical objects across four coarse categories. Across three real-robot FO manipulation tasks, our FOM-SAM3-guided policies demonstrated robustness against distractors, discriminative ability of similar FOs, and extendibility to new FOs.",
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
    paper: "assets/papers/fom_sam3_policy.pdf",
    arxiv: "https://arxiv.org/abs/2609.21621",
    code: "https://github.com/hlmeng-casia/FOM-SAM3-Policy",
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
};
