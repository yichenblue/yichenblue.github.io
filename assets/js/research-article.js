/* Preserve the site's theme toggle, adding keyboard access on opted-in posts. */
(function () {
  "use strict";
  if (!document.body.classList.contains("research-article")) return;
  var themeToggle = document.querySelector('#theme-toggle a[role="button"]');
  if (!themeToggle) return;
  themeToggle.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      themeToggle.click();
    }
  });
})();

/* Citation previews are local UI only: hovering never fetches the linked paper. */
(function () {
  "use strict";
  if (!document.body.classList.contains("research-article")) return;
  var article = document.querySelector(".page__content");
  if (!article) return;

  var chinese = /^zh/i.test(document.documentElement.lang);
  var labels = chinese ? {
    group: "论文链接", copy: "复制链接", copied: "已复制链接",
    failed: "复制失败，请选中网址手动复制"
  } : {
    group: "Paper link", copy: "Copy link", copied: "Link copied",
    failed: "Could not copy. Select the URL to copy it manually."
  };
  var paperHost = /(^|\.)(arxiv\.org|openreview\.net|opt-ml\.org|proceedings\.neurips\.cc|proceedings\.mlr\.press|jmlr\.org|aclanthology\.org|doi\.org)$/i;
  var links = Array.prototype.filter.call(article.querySelectorAll("a[href]"), function (link) {
    if (!link.textContent.trim() || link.querySelector("img, svg")) return false;
    var url = new URL(link.href, document.baseURI);
    return /^https?:$/.test(url.protocol) &&
      (paperHost.test(url.hostname) || link.hasAttribute("data-paper-link"));
  });
  if (!links.length) return;

  function icon(paths) {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + paths + "</svg>";
  }
  var copyIcon = icon('<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/>');
  var panel = document.createElement("span");
  panel.className = "research-link-preview";
  panel.hidden = true;
  panel.setAttribute("role", "group");
  panel.setAttribute("aria-label", labels.group);
  panel.innerHTML = '<span class="research-link-preview__icon">' +
    icon('<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/>') +
    '</span><a class="research-link-preview__url" tabindex="-1"></a>' +
    '<button class="research-link-preview__copy" type="button">' + copyIcon + "</button>";
  var destination = panel.querySelector("a");
  var copy = panel.querySelector("button");
  var announcement = document.createElement("span");
  announcement.className = "research-link-preview__status";
  announcement.setAttribute("role", "status");
  document.body.appendChild(announcement);

  var active = null;
  var showTimer;
  var hideTimer;
  var resetTimer;
  var copyGeneration = 0;
  var touchInput = false;
  var restoringFocus = false;

  function resetCopy() {
    clearTimeout(resetTimer);
    copy.innerHTML = copyIcon;
    copy.title = labels.copy;
    copy.setAttribute("aria-label", labels.copy);
    announcement.textContent = "";
  }

  function hide() {
    copyGeneration++;
    clearTimeout(showTimer);
    clearTimeout(hideTimer);
    if (panel.contains(document.activeElement) && active) {
      restoringFocus = true;
      active.focus({ preventScroll: true });
      restoringFocus = false;
    }
    panel.hidden = true;
    active = null;
    resetCopy();
  }

  function scheduleHide() {
    clearTimeout(showTimer);
    clearTimeout(hideTimer);
    hideTimer = setTimeout(function () {
      var focused = document.activeElement;
      if (active && (active.matches(":hover") || panel.matches(":hover") ||
          focused === active || panel.contains(focused))) return;
      hide();
    }, 180);
  }

  function show(link, point) {
    clearTimeout(showTimer);
    clearTimeout(hideTimer);
    if (active !== link) {
      copyGeneration++;
      resetCopy();
    }
    active = link;
    destination.href = link.href;
    destination.textContent = link.href;
    destination.target = link.target;
    destination.rel = link.rel;
    // Keep the Copy button immediately after its citation in keyboard tab order.
    if (link.nextElementSibling !== panel) link.insertAdjacentElement("afterend", panel);
    panel.hidden = false;
    var rects = Array.prototype.slice.call(link.getClientRects());
    var rect = rects.find(function (r) {
      return point && point.x >= r.left && point.x <= r.right &&
        point.y >= r.top && point.y <= r.bottom;
    }) || rects[rects.length - 1] || link.getBoundingClientRect();
    var bounds = panel.getBoundingClientRect();
    var margin = 12;
    var top = rect.bottom + 6;
    if (top + bounds.height > window.innerHeight - margin) top = rect.top - bounds.height - 6;
    panel.style.left = Math.max(margin, Math.min(rect.left, window.innerWidth - bounds.width - margin)) + "px";
    panel.style.top = Math.max(margin, top) + "px";
  }

  document.addEventListener("pointerdown", function (event) {
    touchInput = event.pointerType === "touch";
    if (active && !panel.contains(event.target) && !active.contains(event.target)) hide();
  }, true);
  document.addEventListener("keydown", function (event) {
    touchInput = false;
    if (event.key === "Escape") hide();
  });
  links.forEach(function (link) {
    link.addEventListener("pointerenter", function (event) {
      if (event.pointerType === "touch") return;
      clearTimeout(hideTimer);
      clearTimeout(showTimer);
      var point = { x: event.clientX, y: event.clientY };
      showTimer = setTimeout(function () { show(link, point); }, 180);
    });
    link.addEventListener("pointerleave", scheduleHide);
    link.addEventListener("focus", function () {
      if (!touchInput && !restoringFocus) show(link);
    });
    link.addEventListener("blur", scheduleHide);
    link.addEventListener("click", hide);
  });
  panel.addEventListener("pointerenter", function () { clearTimeout(hideTimer); });
  panel.addEventListener("pointerleave", scheduleHide);
  panel.addEventListener("focusin", function () { clearTimeout(hideTimer); });
  panel.addEventListener("focusout", scheduleHide);
  window.addEventListener("scroll", function () {
    // Tabbing to an offscreen citation can scroll it into view after focus fires.
    if (active && (document.activeElement === active || panel.contains(document.activeElement))) {
      var rect = active.getBoundingClientRect();
      if (rect.bottom > 0 && rect.top < window.innerHeight) {
        show(active);
        return;
      }
    }
    hide();
  }, { capture: true, passive: true });
  window.addEventListener("resize", hide, { passive: true });

  copy.addEventListener("click", function () {
    var link = active;
    if (!link) return;
    var url = link.href;
    var generation = ++copyGeneration;
    function current() { return active === link && generation === copyGeneration; }
    function report(success) {
      if (!current()) return;
      var message = success ? labels.copied : labels.failed;
      copy.innerHTML = success ? icon('<path d="m5 12 4 4 10-10"/>') : icon('<path d="M12 5v9M12 18v1"/>');
      copy.title = message;
      copy.setAttribute("aria-label", message);
      announcement.textContent = message;
      clearTimeout(resetTimer);
      resetTimer = setTimeout(resetCopy, 2000);
    }
    function fallback() {
      if (!current()) return;
      var input = document.createElement("textarea");
      input.value = url;
      input.className = "research-link-preview__status";
      document.body.appendChild(input);
      input.select();
      var success = false;
      try { success = document.execCommand("copy"); } catch (_) { /* Report the failure below. */ }
      input.remove();
      if (active === link) copy.focus({ preventScroll: true });
      report(success);
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(url).then(function () { report(true); }, fallback);
    } else {
      fallback();
    }
  });
})();
