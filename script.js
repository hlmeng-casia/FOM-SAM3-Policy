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
    video.preload = options.preload || "metadata";
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
    const source = document.createElement("source");
    source.src = item.src;
    source.type = "video/mp4";
    video.append(source, "MP4 playback requires a compatible browser.");
    return video;
  };

  const enableViewportPlayback = (videos) => {
    if (!videos.length) return;
    if ("IntersectionObserver" in window) {
      const playbackObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const playback = entry.target.play();
            if (playback) playback.catch(() => {});
          } else {
            entry.target.pause();
          }
        });
      }, { rootMargin: "120px 0px", threshold: 0.2 });
      videos.forEach((video) => playbackObserver.observe(video));
      return;
    }
    videos.forEach((video) => {
      const playback = video.play();
      if (playback) playback.catch(() => {});
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
      card.append(video);
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
    enableViewportPlayback(cards.map((card) => card.querySelector("video")));
  }

  const mainExperimentVideos = document.querySelector("[data-main-experiment-videos]");
  if (mainExperimentVideos && Array.isArray(data.demoVideos) && data.demoVideos.length) {
    const tasks = [
      { label: "Pick & Place", key: "pick_place" },
      { label: "Push", key: "push" },
      { label: "Assemble", key: "assemble" }
    ];
    const conditions = [
      {
        key: "id",
        label: "In-Distribution",
        description: "The seen target and training scene measure the learned manipulation skill."
      },
      {
        key: "ood_distractors",
        label: "Out-of-Distribution with Distractors",
        description: "A new background and an unrelated object test robustness to ordinary visual interference."
      },
      {
        key: "ood_similar",
        label: "Out-of-Distribution with Similar-FOs",
        description: "A new background and a same-category Similar-FO test queried-identity selection and task completion."
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
    const cards = data.generalizationDemos.map((demo, index) => {
      const stem = `${demo.taskKey}_within_type_new_fos_${demo.methodKey}_${demo.sample}`;
      const item = {
        src: `assets/videos/${stem}.mp4`,
        poster: `assets/videos/${stem}.jpg`,
        title: `${demo.task} · ${demo.method} · New FO ${demo.sample}`
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
      group.textContent = `${demo.task} · New FO ${demo.sample}`;
      const title = document.createElement("strong");
      title.textContent = demo.method;
      caption.append(group, title);
      card.append(video, caption);
      return card;
    });
    generalizationGrid.replaceChildren(...cards);
    enableViewportPlayback(all(".generalization-video-player"));
  }

  const currentYear = document.querySelector("[data-current-year]");
  if (currentYear) currentYear.textContent = new Date().getFullYear();
})();
