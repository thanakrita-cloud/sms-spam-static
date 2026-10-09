/* แกนคำนวณของ SMS Spam Checker (ทำงานในเบราว์เซอร์ ไม่มีเซิร์ฟเวอร์)
   เลียนแบบ scikit-learn: TfidfVectorizer(ngram 1-2, sublinear_tf, L2) + โมเดลเชิงเส้น 3 ตัว
   ทดสอบเทียบกับ Python ด้วย test_core.js */
(function (root) {
  'use strict';
  var MAX_CHARS = 1000, HARD_LIMIT = 5000;

  // เหมือน token_pattern (?u)\b\w\w+\b ของ sklearn = ช่วงตัวอักษร/ตัวเลข/_ ติดกันยาว >= 2
  var TOKEN = /[\p{L}\p{N}_]{2,}/gu;

  function tokens(text) { return (text.toLowerCase().match(TOKEN)) || []; }

  function buildIndex(part) {
    var idx = new Map();
    for (var i = 0; i < part.terms.length; i++) idx.set(part.terms[i], i);
    part._idx = idx;
    return part;
  }

  // คืน {ids: [feature idx], vals: [ค่า TF-IDF หลัง L2]}
  function vectorize(part, text) {
    if (!part._idx) buildIndex(part);
    var toks = tokens(text), grams = toks.slice(), i;
    for (i = 0; i < toks.length - 1; i++) grams.push(toks[i] + ' ' + toks[i + 1]);
    var counts = new Map();
    grams.forEach(function (g) { if (part._idx.has(g)) counts.set(g, (counts.get(g) || 0) + 1); });
    var ids = [], vals = [], norm = 0;
    counts.forEach(function (tf, g) {
      var j = part._idx.get(g);
      var v = (1 + Math.log(tf)) * part.idf[j];      // sublinear_tf
      ids.push(j); vals.push(v); norm += v * v;
    });
    norm = Math.sqrt(norm) || 1;
    for (i = 0; i < vals.length; i++) vals[i] /= norm;
    return { ids: ids, vals: vals };
  }

  function dot(x, w) { var s = 0; for (var i = 0; i < x.ids.length; i++) s += x.vals[i] * w[x.ids[i]]; return s; }

  // Logistic Regression + แยกคะแนนรายคำ (เหมือน explain() ของแอปเดิม)
  function explainLR(M, text, k) {
    var x = vectorize(M.lin, text), rows = [], sum = 0, i;
    for (i = 0; i < x.ids.length; i++) {
      var c = x.vals[i] * M.lin.lr_coef[x.ids[i]];
      sum += c; rows.push([M.lin.terms[x.ids[i]], c]);
    }
    rows.sort(function (a, b) { return Math.abs(b[1]) - Math.abs(a[1]); });
    return { score: sum + M.lin.lr_b, intercept: M.lin.lr_b, rows: rows.slice(0, k || 8), nKnown: x.ids.length };
  }

  function predictSVM(M, text) { return dot(vectorize(M.lin, text), M.lin.svm_coef) + M.lin.svm_b; }
  function predictNB(M, text) { return dot(vectorize(M.nb, text), M.nb.w) + M.nb.b; }

  function hasThai(s) { return /[฀-๿]/.test(s); }
  function mostlyNonLatin(s) {
    var letters = s.match(/\p{L}/gu) || [];
    if (!letters.length) return false;
    var non = letters.filter(function (c) { return !/^[a-z]$/i.test(c); }).length;
    return non / letters.length > 0.5;
  }
  // เหตุผลที่ประเมินไม่ได้ หรือ null
  function scopeIssue(text, nKnown) {
    if (hasThai(text) || mostlyNonLatin(text)) return 'รองรับเฉพาะ SMS ภาษาอังกฤษ';
    if (text.length > HARD_LIMIT) return 'ข้อความยาวเกิน ' + HARD_LIMIT.toLocaleString('en-US') + ' ตัวอักษร และไม่ใช่ SMS สั้นตามขอบเขตของระบบ';
    if (nKnown === 0) return 'ไม่พบคำหรือคู่คำที่โมเดลรู้จัก จึงมีเพียงค่าเริ่มต้นของโมเดล';
    return null;
  }

  var api = { MAX_CHARS: MAX_CHARS, HARD_LIMIT: HARD_LIMIT, TOKEN: TOKEN, tokens: tokens, vectorize: vectorize, explainLR: explainLR,
              predictSVM: predictSVM, predictNB: predictNB, scopeIssue: scopeIssue, hasThai: hasThai, mostlyNonLatin: mostlyNonLatin };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.SpamCore = api;
})(typeof window !== 'undefined' ? window : globalThis);
