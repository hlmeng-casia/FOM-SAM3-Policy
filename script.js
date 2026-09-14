(function () {
  "use strict";
  const data = window.PROJECT_DATA || {};
  const all = (selector) => Array.from(document.querySelectorAll(selector));

  ["title", "venue"].forEach((field) => {
    if (!data[field]) return;
    all(`[data-field="${field}"]`).forEach((node) => (node.textContent = data[field]));
  });

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
    if (!url) {
      link.removeAttribute("href");
      link.setAttribute("aria-disabled", "true");
      link.title = "Coming soon";
      return;
    }
    link.href = url;
    if (!url.startsWith("#")) {
      link.target = "_blank";
      link.rel = "noreferrer";
    }
  });

  const createVideoPlayer = (item, index, className, options = {}) => {
    const video = document.createElement("video");
    video.className = className;
    video.controls = options.controls ?? true;
    video.preload = "none";
    video.playsInline = true;
    video.muted = true;
    video.defaultMuted = true;
    video.volume = 0;
    video.setAttribute("muted", "");
    video.addEventListener("volumechange", () => {
      if (!video.muted) video.muted = true;
      if (video.volume !== 0) video.volume = 0;
    });
    if (options.autoplay) {
      video.autoplay = true;
      video.loop = options.loop ?? true;
    }
    if (item.poster) video.poster = item.poster;
    video.setAttribute("aria-label", item.title || `Robot demonstration ${index + 1}`);
    video.dataset.videoSrc = item.src;
    video.append(document.createTextNode("MP4 playback requires a compatible browser."));
    return video;
  };

  /**
   * 按需连接视频源，避免页面初始化时同时请求全部素材。
   * Attach one video source on demand so the page never requests every asset at startup.
   */
  const loadVideoSource = (video, preload = "metadata") => {
    if (video.dataset.sourceLoaded === "true") {
      if (preload === "auto") video.preload = "auto";
      return;
    }
    const source = document.createElement("source");
    source.src = video.dataset.videoSrc;
    source.type = "video/mp4";
    video.replaceChildren(source, document.createTextNode("MP4 playback requires a compatible browser."));
    video.dataset.sourceLoaded = "true";
    video.preload = preload;
    video.load();
  };

  /**
   * 在浏览器允许时启动静音循环播放，并在媒体可播放后自动重试。
   * Start muted playback and retry as soon as enough media data becomes available.
   */
  const startVideo = (video) => {
    loadVideoSource(video, "auto");
    const retry = () => {
      if (!video.isConnected || video.dataset.inViewport !== "true") return;
      const playback = video.play();
      if (playback) playback.catch(() => {});
    };
    retry();
    if (video.readyState < HTMLMediaElement.HAVE_FUTURE_DATA && video.dataset.retryBound !== "true") {
      video.dataset.retryBound = "true";
      video.addEventListener("canplay", retry);
      video.addEventListener("loadeddata", retry);
    }
  };

  const enableViewportPlayback = (videos) => {
    if (!videos.length) return;
    if ("IntersectionObserver" in window) {
      const sourceObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          loadVideoSource(entry.target, "metadata");
          observer.unobserve(entry.target);
        });
      }, { rootMargin: "260px", threshold: 0.01 });
      const playbackObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.dataset.inViewport = "true";
            startVideo(entry.target);
          } else {
            entry.target.dataset.inViewport = "false";
            entry.target.pause();
          }
        });
      }, { threshold: 0.08 });
      videos.forEach((video) => {
        sourceObserver.observe(video);
        playbackObserver.observe(video);
      });
      return;
    }
    videos.forEach((video) => {
      video.dataset.inViewport = "true";
      startVideo(video);
    });
  };

  const heroVideos = document.querySelector("[data-hero-videos]");
  if (heroVideos && Array.isArray(data.demoVideos) && data.demoVideos.length) {
    const gallery = document.createElement("div");
    gallery.className = "hero-demo-gallery";
    const viewport = document.createElement("div");
    viewport.className = "hero-demo-viewport";
    const track = document.createElement("div");
    track.className = "hero-demo-track";
    track.setAttribute("aria-label", "Robot demonstration video carousel");

    const cards = data.demoVideos.map((item, index) => {
      const card = document.createElement("figure");
      card.className = "hero-demo-card";
      const video = createVideoPlayer(item, index, "hero-demo-player", {
        controls: true,
        autoplay: true,
        loop: true,
        preload: "metadata"
      });
      const caption = document.createElement("figcaption");
      caption.textContent = item.title;
      card.append(video, caption);
      return card;
    });

    const previous = document.createElement("button");
    previous.className = "hero-demo-arrow previous";
    previous.type = "button";
    previous.setAttribute("aria-label", "Previous demonstration");
    const next = document.createElement("button");
    next.className = "hero-demo-arrow next";
    next.type = "button";
    next.setAttribute("aria-label", "Next demonstration");

    const dots = document.createElement("div");
    dots.className = "hero-demo-dots";
    dots.setAttribute("aria-label", "Choose a robot demonstration");
    const dotButtons = cards.map((_card, index) => {
      const dot = document.createElement("button");
      dot.type = "button";
      dot.setAttribute("aria-label", `Go to demonstration ${index + 1}`);
      dot.setAttribute("aria-current", String(index === 0));
      dots.appendChild(dot);
      return dot;
    });

    let activeIndex = 0;
    let carouselAnimation = 0;
    let isAnimating = false;
    const maxScrollLeft = () => Math.max(0, track.scrollWidth - track.clientWidth);
    const cardLeft = (index) => Math.min(cards[index].offsetLeft, maxScrollLeft());
    const updateDots = () => {
      dotButtons.forEach((dot, index) => dot.setAttribute("aria-current", String(index === activeIndex)));
    };
    const syncActiveCard = () => {
      if (isAnimating) return;
      activeIndex = cards.reduce((best, _card, index) => (
        Math.abs(cardLeft(index) - track.scrollLeft) < Math.abs(cardLeft(best) - track.scrollLeft) ? index : best
      ), 0);
      updateDots();
    };
    const stopAnimation = () => {
      window.cancelAnimationFrame(carouselAnimation);
      carouselAnimation = 0;
      isAnimating = false;
      track.classList.remove("is-animating");
    };
    const animateTrackTo = (targetLeft) => {
      window.cancelAnimationFrame(carouselAnimation);
      const startLeft = track.scrollLeft;
      const distance = targetLeft - startLeft;
      if (Math.abs(distance) < 1) {
        stopAnimation();
        track.scrollLeft = targetLeft;
        syncActiveCard();
        return;
      }
      isAnimating = true;
      track.classList.add("is-animating");
      const startedAt = window.performance.now();
      const duration = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 120 : 420;
      const step = (timestamp) => {
        const progress = Math.min(1, (timestamp - startedAt) / duration);
        const eased = 1 - Math.pow(1 - progress, 3);
        track.scrollLeft = startLeft + distance * eased;
        if (progress < 1) {
          carouselAnimation = window.requestAnimationFrame(step);
        } else {
          stopAnimation();
          syncActiveCard();
        }
      };
      carouselAnimation = window.requestAnimationFrame(step);
    };
    const showCard = (index) => {
      const lastStartIndex = cards.findIndex((card) => card.offsetLeft >= maxScrollLeft() - 1);
      const lastIndex = lastStartIndex < 0 ? cards.length - 1 : lastStartIndex;
      activeIndex = Math.max(0, Math.min(lastIndex, index));
      cards.slice(activeIndex, activeIndex + 4).forEach((card) => loadVideoSource(card.querySelector("video")));
      animateTrackTo(cardLeft(activeIndex));
      updateDots();
    };
    previous.addEventListener("click", () => showCard(activeIndex - 1));
    next.addEventListener("click", () => showCard(activeIndex + 1));
    dotButtons.forEach((dot, index) => dot.addEventListener("click", () => showCard(index)));
    let scrollFrame = 0;
    track.addEventListener("scroll", () => {
      window.cancelAnimationFrame(scrollFrame);
      scrollFrame = window.requestAnimationFrame(syncActiveCard);
    }, { passive: true });
    const interruptAnimation = () => {
      stopAnimation();
      syncActiveCard();
    };
    track.addEventListener("pointerdown", interruptAnimation, { passive: true });
    track.addEventListener("wheel", interruptAnimation, { passive: true });
    window.addEventListener("resize", () => {
      stopAnimation();
      track.scrollLeft = cardLeft(activeIndex);
      syncActiveCard();
    });

    track.append(...cards);
    viewport.append(track, previous, next);
    gallery.append(viewport, dots);
    heroVideos.replaceChildren(gallery);
    cards.slice(0, 4).forEach((card) => loadVideoSource(card.querySelector("video")));
    enableViewportPlayback(cards.map((card) => card.querySelector("video")));
  }

  const mainExperimentVideos = document.querySelector("[data-main-experiment-videos]");
  if (mainExperimentVideos && Array.isArray(data.demoVideos) && data.demoVideos.length) {
    const tasks = [
      { label: "Collect Can", key: "pick_place" },
      { label: "Push Box", key: "push" },
      { label: "Lid Cup", key: "assemble" }
    ];
    const conditions = [
      {
        key: "id",
        label: "DemoFO Only",
        description: "The scene contains only the DemoFO used in demonstrations. This setting measures the learned manipulation skill and basic policy capability."
      },
      {
        key: "ood_distractors",
        label: "DemoFO + Distractors",
        description: "Two objects from different coarse categories are added as distractors. This setting evaluates robustness to ordinary visual interference while keeping the target FO unchanged."
      },
      {
        key: "ood_similar",
        label: "DemoFO + SimilarFOs",
        description: "Visually similar objects from the same coarse category are placed beside the DemoFO as confusing distractors. This setting evaluates whether the policy can correctly select the queried DemoFO and complete the manipulation task."
      }
    ];
    const methods = [
      { label: "Ours-DP", key: "ours_dp", ours: true },
      { label: "RGB-DP", key: "rgb_dp", ours: false },
      { label: "Ours-ACT", key: "ours_act", ours: true },
      { label: "RGB-ACT", key: "rgb_act", ours: false }
    ];

    const conditionBlocks = conditions.map((condition, conditionIndex) => {
      const block = document.createElement("section");
      block.className = "experiment-condition-block";
      block.setAttribute("aria-labelledby", `condition-title-${conditionIndex}`);
      const conditionHead = document.createElement("header");
      conditionHead.className = "experiment-condition-head";
      const conditionTitle = document.createElement("h4");
      conditionTitle.id = `condition-title-${conditionIndex}`;
      conditionTitle.textContent = condition.label;
      const conditionDescription = document.createElement("p");
      conditionDescription.textContent = condition.description;
      conditionHead.append(conditionTitle, conditionDescription);
      block.appendChild(conditionHead);
      const conditionContent = document.createElement("div");
      conditionContent.className = "experiment-condition-content";

      tasks.forEach((task, taskIndex) => {
        const taskGroup = document.createElement("section");
        taskGroup.className = "experiment-task-group";
        const taskHead = document.createElement("div");
        taskHead.className = "experiment-task-head";
        const taskTitle = document.createElement("h5");
        taskTitle.textContent = task.label;
        const promptLine = document.createElement("p");
        promptLine.className = "experiment-task-prompt";
        const taskPrompt = data.taskPrompts?.[task.label];
        promptLine.textContent = taskPrompt ? `“${taskPrompt}”` : "";
        if (!taskPrompt) promptLine.setAttribute("aria-hidden", "true");
        taskHead.append(taskTitle, promptLine);

        const scroll = document.createElement("div");
        scroll.className = "experiment-task-scroll";
        const grid = document.createElement("div");
        grid.className = "experiment-method-grid";
        methods.forEach((method, methodIndex) => {
          const slotIndex = conditionIndex * tasks.length * methods.length + taskIndex * methods.length + methodIndex;
          const stem = `${task.key}_${condition.key}_${method.key}_01`;
          const item = {
            src: `assets/videos/${stem}.mp4`,
            poster: `assets/videos/${stem}.jpg`,
            title: `${task.label} · ${condition.label} · ${method.label}`
          };
          const card = document.createElement("article");
          card.className = `experiment-video-card${method.ours ? " ours" : ""}`;
          const heading = document.createElement("div");
          heading.className = "experiment-video-head";
          const methodLabel = document.createElement("strong");
          methodLabel.textContent = method.label;
          heading.appendChild(methodLabel);
          const video = createVideoPlayer(item, slotIndex, "experiment-video-player", {
            autoplay: true,
            controls: false,
            loop: true,
            preload: "metadata"
          });
          card.append(heading, video);
          grid.appendChild(card);
        });
        scroll.appendChild(grid);
        taskGroup.append(taskHead, scroll);
        conditionContent.appendChild(taskGroup);
      });

      block.appendChild(conditionContent);

      return block;
    });
    mainExperimentVideos.replaceChildren(...conditionBlocks);
    enableViewportPlayback(all(".experiment-video-player"));
  }

  const generalizationGrid = document.querySelector("[data-generalization-videos]");
  if (generalizationGrid && Array.isArray(data.generalizationDemos)) {
    const taskNames = [...new Set(data.generalizationDemos.map((demo) => demo.task))];
    const taskGroups = taskNames.map((task) => {
      const section = document.createElement("section");
      section.className = "generalization-task";
      const heading = document.createElement("h4");
      heading.textContent = task;
      const row = document.createElement("div");
      row.className = "generalization-task-grid";
      const taskDemos = data.generalizationDemos.filter((demo) => demo.task === task);
      const cards = taskDemos.map((demo, index) => {
        const item = {
          src: demo.src,
          poster: demo.poster,
          title: `${demo.task} · ${demo.fo} · ${demo.method}`
        };
        const card = document.createElement("article");
        card.className = "generalization-video-card";
        const video = createVideoPlayer(
          item,
          index,
          "generalization-video-player",
          { autoplay: true, controls: false, loop: true, preload: "metadata" }
        );
        const caption = document.createElement("div");
        caption.className = "generalization-video-caption";
        const group = document.createElement("small");
        group.textContent = demo.method;
        const title = document.createElement("strong");
        title.textContent = demo.fo;
        caption.append(group, title);
        card.append(caption, video);
        return card;
      });
      row.append(...cards);
      section.append(heading, row);
      return section;
    });
    generalizationGrid.replaceChildren(...taskGroups);
    enableViewportPlayback(all(".generalization-video-player"));
  }

  const sceneGeneralizationGrid = document.querySelector("[data-scene-generalization-videos]");
  if (sceneGeneralizationGrid && Array.isArray(data.sceneGeneralizationDemos)) {
    const cards = data.sceneGeneralizationDemos.map((demo, index) => {
      const item = { src: demo.src, title: `${demo.setting} · ${demo.fo}` };
      const card = document.createElement("article");
      card.className = "generalization-video-card";
      const video = createVideoPlayer(
        item,
        index,
        "generalization-video-player",
        { autoplay: true, controls: false, loop: true, preload: "metadata" }
      );
      const caption = document.createElement("div");
      caption.className = "generalization-video-caption";
      const setting = document.createElement("small");
      setting.textContent = demo.setting;
      const title = document.createElement("strong");
      title.textContent = demo.fo;
      caption.append(setting, title);
      card.append(caption, video);
      return card;
    });
    sceneGeneralizationGrid.replaceChildren(...cards);
    enableViewportPlayback(all(".scene-generalization-grid .generalization-video-player"));
  }

  const currentYear = document.querySelector("[data-current-year]");
  if (currentYear) currentYear.textContent = new Date().getFullYear();
})();
