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
