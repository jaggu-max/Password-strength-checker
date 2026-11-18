(function () {
  "use strict";

  var input = document.getElementById("login-password");
  var toggle = document.getElementById("login-toggle");
  if (!input || !toggle) return;

  toggle.addEventListener("click", function () {
    var show = input.type === "password";
    input.type = show ? "text" : "password";
    toggle.textContent = show ? "Hide" : "Show";
    toggle.setAttribute("aria-pressed", String(show));
    toggle.setAttribute("aria-label", show ? "Hide password" : "Show password");
    input.focus();
  });
})();
