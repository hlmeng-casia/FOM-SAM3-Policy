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
  if (heroVideos && Array.isArray(data.demoVideos)) {
    const previews = data.demoVideos.slice(0, 3).map((item, index) => {
      const card = document.createElement("figure");
      card.className = "hero-video-card";
      const caption = document.createElement("figcaption");
      caption.innerHTML = `<span>Demonstration ${String(index + 1).padStart(2, "0")}</span><small>Robot rollout</small>`;
      card.append(createVideoPlayer(item, index, "hero-video-player", {
        autoplay: true,
        controls: false,
        loop: true,
        preload: "auto"
      }), caption);
      return card;
    });
    heroVideos.replaceChildren(...previews);
    heroVideos.querySelectorAll("video").forEach((video) => {
      const playback = video.play();
      if (playback) playback.catch(() => {});
    });
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
    const methodGroups = [
      { names: ["Ours-DP", "RGB-DP"] },
      { names: ["Ours-ACT", "RGB-ACT"] }
    ];
    const tabs = document.createElement("div");
    tabs.className = "experiment-condition-tabs";
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", "Evaluation condition");

    const panels = conditions.map((condition, conditionIndex) => {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.id = `condition-tab-${conditionIndex}`;
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-controls", `condition-panel-${conditionIndex}`);
      tab.setAttribute("aria-selected", String(conditionIndex === 0));
      tab.textContent = condition.label;
      tabs.appendChild(tab);

      const panel = document.createElement("div");
      panel.className = "experiment-condition-panel";
      panel.id = `condition-panel-${conditionIndex}`;
      panel.setAttribute("role", "tabpanel");
      panel.setAttribute("aria-labelledby", tab.id);
      panel.hidden = conditionIndex !== 0;

      taskNames.forEach((task, taskIndex) => {
        const taskGroup = document.createElement("section");
        taskGroup.className = "experiment-task-group";
        const taskHead = document.createElement("div");
        taskHead.className = "experiment-task-head";
        taskHead.innerHTML = `<h4>${task}</h4><small>${condition.label}</small>`;

        const scroll = document.createElement("div");
        scroll.className = "experiment-task-scroll";
        methodGroups.forEach((group) => {
          const row = document.createElement("div");
          row.className = "experiment-policy-row";
          const grid = document.createElement("div");
          grid.className = `experiment-method-grid${group.names.length === 2 ? " two-items" : ""}`;
          group.names.forEach((methodName) => {
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
          row.appendChild(grid);
          scroll.appendChild(row);
        });
        taskGroup.append(taskHead, scroll);
        panel.appendChild(taskGroup);
      });

      tab.addEventListener("click", () => {
        Array.from(tabs.children).forEach((button) => button.setAttribute("aria-selected", String(button === tab)));
        panels.forEach((item) => (item.hidden = item !== panel));
      });
      return panel;
    });
    mainExperimentVideos.replaceChildren(tabs, ...panels);
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

  const robotBody = document.querySelector("[data-robot-results]");
  if (robotBody && Array.isArray(data.robotResults)) {
    robotBody.replaceChildren(...data.robotResults.map((row) => {
      const tr = document.createElement("tr");
      if (row.ours) tr.className = "ours-row";
      [row.method, ...row.id, ...row.distractors, ...row.similar].forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        tr.appendChild(cell);
      });
      return tr;
    }));
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
