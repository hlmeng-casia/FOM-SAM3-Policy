(function () {
  "use strict";
  const data = window.PROJECT_DATA || {};
  const all = (selector) => Array.from(document.querySelectorAll(selector));

  ["shortName", "title", "label", "summary"].forEach((field) => {
    if (!data[field]) return;
    all(`[data-field="${field}"]`).forEach((node) => (node.textContent = data[field]));
  });

  const authors = document.querySelector("[data-authors]");
  const institutions = document.querySelector("[data-institutions]");
  if (authors && Array.isArray(data.authors)) authors.textContent = data.authors.join(" · ");
  if (institutions && Array.isArray(data.institutions)) institutions.textContent = data.institutions.join(" · ");

  const abstract = document.querySelector("[data-abstract]");
  if (abstract && Array.isArray(data.abstract)) {
    const highlights = Array.isArray(data.abstractHighlights) ? data.abstractHighlights : [];
    const escapedHighlights = highlights.map((phrase) => phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const highlightPattern = escapedHighlights.length ? new RegExp(`(${escapedHighlights.join("|")})`, "g") : null;
    abstract.replaceChildren(...data.abstract.map((text) => {
      const paragraph = document.createElement("p");
      if (!highlightPattern) {
        paragraph.textContent = text;
        return paragraph;
      }
      text.split(highlightPattern).forEach((part) => {
        if (!part) return;
        if (highlights.includes(part)) {
          const strong = document.createElement("strong");
          strong.textContent = part;
          paragraph.append(strong);
        } else {
          paragraph.append(document.createTextNode(part));
        }
      });
      return paragraph;
    }));
  }

  all("[data-resource]").forEach((link) => {
    const url = data.links?.[link.dataset.resource];
    if (url) {
      link.href = url;
      if (!url.startsWith("#")) {
        link.target = "_blank";
        link.rel = "noreferrer";
      }
    } else {
      link.classList.add("disabled");
      link.setAttribute("aria-disabled", "true");
      link.removeAttribute("href");
    }
  });

  const createVideoPlayer = (item, index, className, options = {}) => {
    const video = document.createElement("video");
    video.className = className;
    video.controls = options.controls ?? true;
    video.preload = options.preload || "metadata";
    video.playsInline = true;
    if (options.autoplay) {
      video.autoplay = true;
      video.muted = true;
      video.defaultMuted = true;
      video.loop = options.loop ?? true;
      video.setAttribute("muted", "");
    }
    if (item.poster) video.poster = item.poster;
    video.setAttribute("aria-label", item.title || `Robot demonstration ${index + 1}`);
    const source = document.createElement("source");
    source.src = item.src;
    source.type = "video/mp4";
    video.append(source, "MP4 playback requires a compatible browser.");
    return video;
  };

  const heroVideos = document.querySelector("[data-hero-videos]");
  if (heroVideos && Array.isArray(data.demoVideos) && data.demoVideos.length) {
    const featuredVideo = createVideoPlayer(data.demoVideos[0], 0, "hero-featured-player", {
      autoplay: true,
      controls: true,
      loop: true,
      preload: "auto"
    });

    const gallery = document.createElement("div");
    gallery.className = "hero-demo-gallery";
    const galleryHead = document.createElement("div");
    galleryHead.className = "hero-demo-gallery-head";
    galleryHead.innerHTML = `<strong>Demo gallery</strong><span>${data.demoVideos.length} rollouts</span>`;
    const track = document.createElement("div");
    track.className = "hero-demo-track";
    track.setAttribute("aria-label", "Select a featured robot demonstration");

    const slots = data.demoVideos.map((item, index) => {
      const slot = document.createElement("button");
      slot.className = "hero-demo-slot";
      slot.type = "button";
      slot.setAttribute("aria-label", `Show ${item.title || `robot demonstration ${index + 1}`}`);
      slot.setAttribute("aria-current", String(index === 0));
      const poster = document.createElement("img");
      poster.src = item.poster || "";
      poster.alt = "";
      poster.loading = "lazy";
      const label = document.createElement("span");
      label.textContent = item.title || `Robot demonstration ${String(index + 1).padStart(2, "0")}`;
      slot.append(poster, label);
      slot.addEventListener("click", () => {
        const source = featuredVideo.querySelector("source");
        featuredVideo.pause();
        if (source) source.src = item.src;
        featuredVideo.poster = item.poster || "";
        featuredVideo.setAttribute("aria-label", item.title || `Robot demonstration ${index + 1}`);
        slots.forEach((candidate) => candidate.setAttribute("aria-current", String(candidate === slot)));
        featuredVideo.load();
        const playback = featuredVideo.play();
        if (playback) playback.catch(() => {});
      });
      return slot;
    });

    track.append(...slots);
    gallery.append(galleryHead, track);
    heroVideos.replaceChildren(featuredVideo, gallery);
    const playback = featuredVideo.play();
    if (playback) playback.catch(() => {});
  }

  const mainExperimentVideos = document.querySelector("[data-main-experiment-videos]");
  if (mainExperimentVideos && Array.isArray(data.demoVideos) && data.demoVideos.length) {
    const taskNames = ["Pick & Place", "Push", "Assemble"];
    const conditions = [
      { key: "id", label: "ID" },
      { key: "distractors", label: "OOD-Distractors" },
      { key: "similar", label: "OOD-Similar" }
    ];
    const methods = Array.isArray(data.robotResults) ? data.robotResults : [];
    const methodOrder = ["Ours-DP", "RGB-DP", "Ours-ACT", "RGB-ACT"];

    const conditionBlocks = conditions.map((condition, conditionIndex) => {
      const block = document.createElement("section");
      block.className = "experiment-condition-block";
      block.setAttribute("aria-labelledby", `condition-title-${conditionIndex}`);
      const conditionHead = document.createElement("header");
      conditionHead.className = "experiment-condition-head";
      const conditionTitle = document.createElement("h4");
      conditionTitle.id = `condition-title-${conditionIndex}`;
      conditionTitle.textContent = condition.label;
      conditionHead.appendChild(conditionTitle);
      block.appendChild(conditionHead);

      taskNames.forEach((task, taskIndex) => {
        const taskGroup = document.createElement("section");
        taskGroup.className = "experiment-task-group";
        const taskHead = document.createElement("div");
        taskHead.className = "experiment-task-head";
        taskHead.innerHTML = `<h5>${task}</h5>`;

        const scroll = document.createElement("div");
        scroll.className = "experiment-task-scroll";
        const grid = document.createElement("div");
        grid.className = "experiment-method-grid";
        methodOrder.forEach((methodName) => {
          const methodIndex = methods.findIndex((method) => method.method === methodName);
          if (methodIndex < 0) return;
          const method = methods[methodIndex];
          const slotIndex = conditionIndex * taskNames.length * methods.length + taskIndex * methods.length + methodIndex;
          const item = data.demoVideos[slotIndex % data.demoVideos.length];
          const card = document.createElement("article");
          card.className = `experiment-video-card${method.ours ? " ours" : ""}`;
          const heading = document.createElement("div");
          heading.className = "experiment-video-head";
          heading.innerHTML = `<strong>${method.method}</strong><output>${method[condition.key]?.[taskIndex] || "--"}</output>`;
          const video = createVideoPlayer(item, slotIndex, "experiment-video-player", { preload: "none" });
          card.append(heading, video);
          grid.appendChild(card);
        });
        scroll.appendChild(grid);
        taskGroup.append(taskHead, scroll);
        block.appendChild(taskGroup);
      });

      return block;
    });
    mainExperimentVideos.replaceChildren(...conditionBlocks);
  }

  const videoGallery = document.querySelector("[data-video-gallery]");
  if (videoGallery && Array.isArray(data.demoVideos)) {
    const galleryVideos = Array.from({ length: 18 }, (_, index) => data.demoVideos[index % data.demoVideos.length]);
    const cards = galleryVideos.map((item, index) => {
      const card = document.createElement("article");
      card.className = "media-card video-card";

      const video = createVideoPlayer(item, index, "media-player");

      const details = document.createElement("div");
      const status = document.createElement("small");
      status.textContent = "Robot rollout";
      const title = document.createElement("h3");
      title.textContent = `Robot demonstration ${String(index + 1).padStart(2, "0")}`;
      const description = document.createElement("p");
      description.textContent = "Robot manipulation rollout.";
      details.append(status, title, description);
      card.append(video, details);
      return card;
    });
    videoGallery.replaceChildren(...cards);
  }

  const citation = document.querySelector("[data-citation]");
  if (citation && data.citation) citation.textContent = data.citation;

  const copyButton = document.querySelector("[data-copy-citation]");
  if (copyButton && citation) {
    copyButton.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(citation.textContent.trim());
        copyButton.textContent = "Copied";
      } catch (_error) {
        copyButton.textContent = "Select and copy";
      }
      window.setTimeout(() => (copyButton.textContent = "Copy citation"), 1600);
    });
  }

  const menuButton = document.querySelector("[data-menu-button]");
  const nav = document.querySelector("[data-nav]");
  if (menuButton && nav) {
    const closeMenu = () => {
      menuButton.setAttribute("aria-expanded", "false");
      nav.classList.remove("open");
    };
    menuButton.addEventListener("click", () => {
      const open = menuButton.getAttribute("aria-expanded") !== "true";
      menuButton.setAttribute("aria-expanded", String(open));
      nav.classList.toggle("open", open);
    });
    all("a", nav).forEach((link) => link.addEventListener("click", closeMenu));
    window.addEventListener("resize", () => {
      if (window.innerWidth > 680) closeMenu();
    });
  }

  const currentYear = document.querySelector("[data-current-year]");
  if (currentYear) currentYear.textContent = new Date().getFullYear();
})();
