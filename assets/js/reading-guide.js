(function () {
  "use strict";

  function initializeReadingGuide() {
    var guide = document.querySelector(".reading-layout .reading-guide");
    if (!guide) return;

    var disclosure = guide.querySelector("details");
    var summary = guide.querySelector("summary");
    var desktop = window.matchMedia("(min-width: 1024px)");
    var items = Array.from(guide.querySelectorAll('a[href^="#"]')).map(function (link) {
      return { link: link, heading: document.getElementById(link.hash.slice(1)) };
    }).filter(function (item) { return item.heading; });

    function updateDisclosure() {
      disclosure.open = desktop.matches;
      if (desktop.matches) {
        summary.setAttribute("aria-disabled", "true");
        summary.setAttribute("tabindex", "-1");
      } else {
        summary.removeAttribute("aria-disabled");
        summary.removeAttribute("tabindex");
      }
    }

    summary.addEventListener("click", function (event) {
      if (desktop.matches) event.preventDefault();
    });
    if (desktop.addEventListener) desktop.addEventListener("change", updateDisclosure);
    else desktop.addListener(updateDisclosure);
    updateDisclosure();

    var scheduled = false;
    function markCurrentSection() {
      scheduled = false;
      var current = items[0];
      items.forEach(function (item) {
        if (item.heading.getBoundingClientRect().top <= 115) current = item;
      });
      items.forEach(function (item) {
        if (item === current) item.link.setAttribute("aria-current", "location");
        else item.link.removeAttribute("aria-current");
      });
    }
    function scheduleUpdate() {
      if (!scheduled) {
        scheduled = true;
        window.requestAnimationFrame(markCurrentSection);
      }
    }

    // Handle these anchors before the theme's general smooth-scroll handler.
    guide.addEventListener("click", function (event) {
      var link = event.target.closest('a[href^="#"]');
      if (!link || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      var heading = document.getElementById(link.hash.slice(1));
      if (!heading) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      if (!desktop.matches) disclosure.open = false;
      var top = window.scrollY + heading.getBoundingClientRect().top - 90;
      window.history.pushState(null, "", link.hash);
      window.scrollTo({ top: Math.max(0, top), behavior: "auto" });
      heading.setAttribute("tabindex", "-1");
      heading.focus({ preventScroll: true });
      scheduleUpdate();
    }, true);

    window.addEventListener("scroll", scheduleUpdate, { passive: true });
    window.addEventListener("resize", scheduleUpdate);
    window.addEventListener("hashchange", scheduleUpdate);
    window.addEventListener("load", scheduleUpdate);
    document.querySelectorAll(".page__content details").forEach(function (detail) {
      detail.addEventListener("toggle", scheduleUpdate);
    });
    markCurrentSection();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeReadingGuide);
  } else {
    initializeReadingGuide();
  }
}());
