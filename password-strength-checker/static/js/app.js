/* ML Password Strength Checker – frontend logic.
   The password is only held in the input box and sent once per click to
   POST /api/check. It is never stored in localStorage or logged. */

(function () {
  "use strict";

  var MAX_LEN = 128;

  var els = {
    input: document.getElementById("password"),
    toggle: document.getElementById("toggle"),
    check: document.getElementById("check"),
    clear: document.getElementById("clear"),
    error: document.getElementById("input-error"),
    result: document.getElementById("result"),
    badge: document.getElementById("badge"),
    confidence: document.getElementById("confidence"),
    meter: document.getElementById("meter"),
    analysis: document.getElementById("analysis"),
    suggestions: document.getElementById("suggestions"),
    modelUsed: document.getElementById("model-used")
  };

  /* ---------- small DOM helpers (textContent only -> no HTML injection) ---------- */
  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function clearNode(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function pct(x, digits) { return (x * 100).toFixed(digits === undefined ? 2 : digits) + "%"; }
  function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

  /* ---------- input behaviour ---------- */
  function showError(message) {
    els.error.textContent = message;
    els.error.hidden = false;
    els.input.classList.add("invalid");
    els.input.setAttribute("aria-invalid", "true");
  }
  function hideError() {
    els.error.hidden = true;
    els.error.textContent = "";
    els.input.classList.remove("invalid");
    els.input.removeAttribute("aria-invalid");
  }

  els.toggle.addEventListener("click", function () {
    var show = els.input.type === "password";
    els.input.type = show ? "text" : "password";
    els.toggle.textContent = show ? "Hide" : "Show";
    els.toggle.setAttribute("aria-pressed", String(show));
    els.toggle.setAttribute("aria-label", show ? "Hide password" : "Show password");
    els.input.focus();
  });

  els.input.addEventListener("input", function () {
    hideError();
    els.result.hidden = true;           // old result no longer matches the text
  });

  els.input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); runCheck(); }
  });

  els.clear.addEventListener("click", function () {
    els.input.value = "";
    els.input.type = "password";
    els.toggle.textContent = "Show";
    els.toggle.setAttribute("aria-pressed", "false");
    els.toggle.setAttribute("aria-label", "Show password");
    hideError();
    els.result.hidden = true;
    els.input.focus();
  });

  els.check.addEventListener("click", runCheck);

  /* ---------- API call ---------- */
  function setLoading(isLoading) {
    els.check.disabled = isLoading;
    els.check.textContent = isLoading ? "Checking\u2026" : "Check Password Strength";
  }

  function runCheck() {
    var pw = els.input.value;
    hideError();

    if (pw.length === 0) { showError("Please enter a password to check."); els.input.focus(); return; }
    if (pw.length > MAX_LEN) { showError("Password is too long (maximum " + MAX_LEN + " characters)."); return; }

    setLoading(true);
    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, 10000);

    fetch("/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: pw }),
      cache: "no-store",
      credentials: "same-origin",
      signal: controller.signal
    })
      .then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (data) {
          return { ok: res.ok, data: data };
        });
      })
      .then(function (r) {
        if (!r.ok) { showError(r.data.error || "The server could not process the request."); return; }
        renderResult(r.data);
      })
      .catch(function (err) {
        showError(err && err.name === "AbortError"
          ? "The request timed out. Please try again."
          : "Could not reach the server. Make sure the Flask app is running.");
      })
      .then(function () {
        clearTimeout(timer);
        setLoading(false);
      });
  }

  /* ---------- render result ---------- */
  function chip(label, value, bad) {
    var c = el("div", "chip" + (bad ? " flag-bad" : ""));
    c.appendChild(el("span", "k", label));
    c.appendChild(el("span", "v", String(value)));
    return c;
  }

  function renderResult(d) {
    var level = d.category;
    els.badge.textContent = cap(level);
    els.badge.setAttribute("data-level", level);
    els.meter.setAttribute("data-level", level);
    els.meter.setAttribute("aria-label", "Strength indicator: " + cap(level));
    els.confidence.textContent = pct(d.confidence, 1);

    var a = d.analysis;
    clearNode(els.analysis);
    els.analysis.appendChild(chip("Length", a.length));
    els.analysis.appendChild(chip("Uppercase", a.uppercase));
    els.analysis.appendChild(chip("Lowercase", a.lowercase));
    els.analysis.appendChild(chip("Numbers", a.digits));
    els.analysis.appendChild(chip("Special chars", a.symbols));
    els.analysis.appendChild(chip("Repeated chars", a.repeated_chars));
    els.analysis.appendChild(chip("Entropy (bits/char)", a.entropy_bits_per_char));
    els.analysis.appendChild(chip("Common password", a.common_password ? "Yes" : "No", a.common_password));
    els.analysis.appendChild(chip("Dictionary word", a.dictionary_word ? "Found" : "None", a.dictionary_word));

    clearNode(els.suggestions);
    d.suggestions.forEach(function (s) { els.suggestions.appendChild(el("li", "", s)); });

    els.modelUsed.textContent = "Classified by: " + d.model;
    els.result.hidden = false;
    els.result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  /* ==========================================================================
     Model results section (reads /api/model-info -> models/metrics.json)
     ========================================================================== */
  function loadModelInfo() {
    fetch("/api/model-info", { cache: "no-store" })
      .then(function (res) {
        if (!res.ok) throw new Error("no metrics");
        return res.json();
      })
      .then(renderModelInfo)
      .catch(function () {
        var box = document.getElementById("results-error");
        box.textContent = "Model metrics are not available yet. Run python ml/train_model.py and refresh this page.";
        box.hidden = false;
      });
  }

  function renderModelInfo(m) {
    document.getElementById("results-body").hidden = false;

    /* summary stats */
    var stats = document.getElementById("stats");
    clearNode(stats);
    [
      ["Records used", m.dataset.total_records_used.toLocaleString()],
      ["Features per password", m.dataset.n_features],
      ["Train / Val / Test", m.split.train.toLocaleString() + " / " + m.split.validation.toLocaleString() + " / " + m.split.test.toLocaleString()],
      ["Selected model", m.best_model]
    ].forEach(function (s) {
      var box = el("div", "stat");
      box.appendChild(el("span", "k", s[0]));
      box.appendChild(el("span", "v", String(s[1])));
      stats.appendChild(box);
    });

    /* comparison table */
    var tbody = document.querySelector("#compare-table tbody");
    clearNode(tbody);
    Object.keys(m.models).forEach(function (name) {
      var t = m.models[name].test;
      var tr = el("tr", name === m.best_model ? "best" : "");
      var first = el("td", "", name);
      if (name === m.best_model) first.appendChild(el("span", "tag", "Selected"));
      tr.appendChild(first);
      tr.appendChild(el("td", "", pct(t.accuracy)));
      tr.appendChild(el("td", "", pct(t.precision_weighted)));
      tr.appendChild(el("td", "", pct(t.recall_weighted)));
      tr.appendChild(el("td", "", pct(t.f1_weighted)));
      tr.appendChild(el("td", "", pct(m.models[name].validation.f1_weighted)));
      tbody.appendChild(tr);
    });

    /* best-model note */
    var best = document.getElementById("best-note");
    clearNode(best);
    var p1 = el("p", "", "");
    p1.appendChild(el("strong", "", "Selected model: " + m.best_model + ". "));
    p1.appendChild(document.createTextNode(
      "Chosen automatically using the rule \u201C" + m.selection_rule + "\u201D. " +
      "The test set was kept separate and is used only for the figures reported here."));
    best.appendChild(p1);

    /* confusion matrices */
    var wrap = document.getElementById("matrices");
    clearNode(wrap);
    Object.keys(m.models).forEach(function (name) {
      var cm = m.models[name].test.confusion_matrix;
      var max = Math.max.apply(null, [].concat.apply([], cm)) || 1;
      var card = el("div", "card mini matrix-card");
      card.appendChild(el("h4", "", name));
      var table = el("table", "matrix");
      var head = el("tr");
      head.appendChild(el("th", "corner", "actual \\ pred."));
      m.class_names.forEach(function (c) { head.appendChild(el("th", "", cap(c))); });
      table.appendChild(head);
      cm.forEach(function (row, i) {
        var tr = el("tr");
        tr.appendChild(el("th", "", cap(m.class_names[i])));
        row.forEach(function (v) {
          var td = el("td", "", v.toLocaleString());
          td.style.backgroundColor = "rgba(26, 60, 43," + (0.06 + 0.45 * (v / max)).toFixed(2) + ")";
          tr.appendChild(td);
        });
        table.appendChild(tr);
      });
      card.appendChild(table);
      wrap.appendChild(card);
    });

    /* feature importances */
    if (m.feature_importances && m.feature_importances.length) {
      var imp = document.getElementById("importance");
      clearNode(imp);
      var top = m.feature_importances[0].importance || 1;
      m.feature_importances.forEach(function (f) {
        var row = el("div", "bar-row");
        row.appendChild(el("span", "", f.feature.replace(/_/g, " ")));
        var track = el("div", "bar-track");
        var fill = el("div", "bar-fill");
        fill.style.width = Math.max(2, (f.importance / top) * 100).toFixed(1) + "%";
        track.appendChild(fill);
        row.appendChild(track);
        row.appendChild(el("span", "bar-val", pct(f.importance, 1)));
        imp.appendChild(row);
      });
      document.getElementById("importance-wrap").hidden = false;
    }

    /* dataset note: only states what the metrics file proves */
    var note = document.getElementById("dataset-note");
    clearNode(note);
    var cc = m.dataset.class_counts;
    var p = el("p", "", "");
    p.appendChild(el("strong", "", "Dataset: "));
    p.appendChild(document.createTextNode(
      m.dataset.file + " \u2013 " + cc.weak.toLocaleString() + " weak, " +
      cc.medium.toLocaleString() + " medium and " + cc.strong.toLocaleString() +
      " strong unique passwords. " +
      (m.smote_applied_to_training_only
        ? "SMOTE class balancing was applied to the training data only; validation and test data keep their real class distribution."
        : "SMOTE was not applied in this training run.")));
    note.appendChild(p);
    if (m.dataset.synthetic_generated) {
      var p2 = el("p", "", "");
      p2.appendChild(el("strong", "", "Please note: "));
      p2.appendChild(document.createTextNode(
        "this bundled dataset is generated by ml/generate_dataset.py and labelled by a documented scoring policy. " +
        "The scores therefore show how well each model learns that policy, not how well a password resists real attackers. " +
        "Train on a real labelled dataset (see README) to evaluate on real-world data."));
      note.appendChild(p2);
    }
  }

  loadModelInfo();
})();
