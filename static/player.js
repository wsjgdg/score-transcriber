// 共享：音频播放器（播放/暂停 + 可拖动进度条）与结果卡片渲染
// 被 app.html（转谱后）与 view.html（历史查看）复用。
(function () {
  function fmt(s) {
    s = Math.max(0, Math.floor(s || 0));
    var m = Math.floor(s / 60), ss = s % 60;
    return m + ":" + (ss < 10 ? "0" : "") + ss;
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  // 绑定一个播放器：audio 元素 + 播放按钮 + range 进度条 + 时间标签
  window.initPlayer = function (audio, playBtn, progress, timeLabel) {
    if (!audio || !playBtn || !progress) return;
    playBtn.addEventListener("click", function () {
      if (audio.paused) audio.play(); else audio.pause();
    });
    audio.addEventListener("play", function () { playBtn.textContent = "⏸ 暂停"; });
    audio.addEventListener("pause", function () { playBtn.textContent = "▶ 播放"; });
    audio.addEventListener("ended", function () { playBtn.textContent = "▶ 播放"; });
    audio.addEventListener("loadedmetadata", function () {
      if (timeLabel) timeLabel.textContent = "0:00 / " + fmt(audio.duration);
    });
    audio.addEventListener("timeupdate", function () {
      if (audio.duration) {
        progress.value = (audio.currentTime / audio.duration) * 100;
        if (timeLabel) timeLabel.textContent = fmt(audio.currentTime) + " / " + fmt(audio.duration);
      }
    });
    progress.addEventListener("input", function () {
      if (audio.duration) audio.currentTime = (progress.value / 100) * audio.duration;
    });
  };

  // 把一条记录（转谱响应或历史记录，结构一致）渲染进 #result 容器
  window.renderResult = function (rec) {
    var box = document.getElementById("result");
    if (!box) return;
    var s = rec.stats || {};
    var conf = (s.confidence != null) ? s.confidence : "—";
    var cmed = (s.confidence_median != null) ? s.confidence_median : "—";
    var chigh = (s.high_conf_ratio != null) ? s.high_conf_ratio : "—";
    var extra = [];
    if (s.denoise_threshold && s.denoise_threshold > 0)
      extra.push(["降噪阈值", s.denoise_threshold.toFixed(2)]);
    var statsHtml = [
      ["音符数", s.num_notes], ["和弦事件", s.num_events], ["时长(秒)", s.duration_sec],
      ["BPM", s.bpm + (s.bpm_estimated ? " *" : "")], ["拍号", s.beats_per_bar + "/" + (s.beats_denom || 4)],
      ["响应强度(均值)", conf + "%"], ["响应强度(中值)", cmed + "%"], ["强响应占比", chigh + "%"]
    ].concat(extra).map(function (x) {
      return '<div class="stat"><div class="k">' + x[0] + '</div><div class="v">' + x[1] + '</div></div>';
    }).join("");

    var f = rec.files || {};
    var hasAudio = !!f.audio;
    var hasScore = !!f.score_audio;

    var tip = "";
    var c = (s.confidence != null) ? s.confidence : 0;
    if (s.denoised_count && s.denoised_count > 0)
      tip += "已启用降噪模式，过滤了 " + s.denoised_count + " 个弱响应音符。";
    if (s.separate_error) tip += " 伴奏分离未生效（" + s.separate_error + "），已回退原音频转录。";
    if (s.high_conf_ratio != null) {
      if (s.high_conf_ratio >= 70) tip += " 强响应音符占 " + s.high_conf_ratio + "%，模型对这些音符的激活较充分。";
      else if (s.high_conf_ratio < 40) tip += " 强响应音符仅占 " + s.high_conf_ratio + "%，较多音符属弱响应（偏猜测），建议开降噪或分离伴奏后重试。";
    }
    if (s.high_conf_empty) tip += " 强响应过滤后无剩余音符，已回退保留全部结果（本段音频整体响应偏弱，建议开降噪或分离伴奏）。";
    if (c >= 75) tip += " 模型对音符的响应强度较高，检出结果参考价值较高。";
    else if (c >= 55) tip += " 响应强度中等，建议对照钢琴卷帘图人工核对起音与音高。";
    else if (c > 0) tip += " 响应强度偏低，可能伴奏较强或录音嘈杂；此数值代表模型激活强度，并非转录准确度保证，建议开启降噪模式或换主旋律更清晰的片段。";

    // 两个播放器并排对比；两者都有时提供“对比播放”按钮
    var compareHtml = "";
    if (hasAudio || hasScore) {
      compareHtml =
        (hasAudio && hasScore ?
          '<button id="playBoth" class="go" style="margin:0 0 10px;min-height:40px;padding:8px 16px">▶ 对比播放（原音频 + 钢琴谱）</button>' : '') +
        '<div class="compare-cols">' +
          (hasAudio ?
            '<div class="pcol"><h4>原音频</h4>' +
            '<div class="player">' +
              '<button id="playOrig" class="go" style="margin:0;min-height:40px;padding:8px 14px">▶ 播放</button>' +
              '<input id="progOrig" type="range" min="0" max="100" value="0" style="flex:1" />' +
              '<span id="timeOrig" style="font-size:12px;color:var(--muted);min-width:92px;text-align:right">0:00 / 0:00</span>' +
            '</div><audio id="audioOrig" style="display:none"></audio></div>' : '') +
          (hasScore ?
            '<div class="pcol"><h4>识别钢琴谱</h4>' +
            '<div class="player">' +
              '<button id="playScore" class="go" style="margin:0;min-height:40px;padding:8px 14px">▶ 播放</button>' +
              '<input id="progScore" type="range" min="0" max="100" value="0" style="flex:1" />' +
              '<span id="timeScore" style="font-size:12px;color:var(--muted);min-width:92px;text-align:right">0:00 / 0:00</span>' +
            '</div><audio id="audioScore" style="display:none"></audio></div>' : '') +
        '</div>';
    }

    // 本次启用的优化选项标记
    var optTags = [];
    if (s.key && s.key !== "C") optTags.push("选调 " + (s.key || "C"));
    if (s.separated) optTags.push("已分离伴奏(" + (s.separate_target || "") + ")");
    if (s.high_conf_only) optTags.push("仅高置信音符");
    if (s.auto_bpm) optTags.push("自动估计BPM");
    if (s.smart_denoise) optTags.push("智能降噪");
    if (s.lyrics_source) optTags.push("歌词(" + (s.lyrics_source === "asr" ? "ASR识别" : "手动填词") + (s.lyrics_count ? "·" + s.lyrics_count + "字" : "") + ")");
    if (s.hpss) optTags.push("HPSS鼓点分离");
    if (s.melody) optTags.push("人声单旋律模式");
    if (s.harmonic_filter) optTags.push("纯净后处理(泛音过滤+最小音长+间隙合并+onset确认)");
    if (s.rms_vel) optTags.push("RMS力度");
    if (s.fine_quant) optTags.push("细网格量化");
    if (s.auto_beats) optTags.push("自动拍号" + (s.beats_estimated ? "(" + s.beats_per_bar + "/" + (s.beats_denom || 4) + ")" : ""));
    if (s.octave_shift) optTags.push("八度偏移 " + (s.octave_shift > 0 ? "+" : "") + s.octave_shift);
    if (s.f0_octave_fix) optTags.push(s.f0_octave_shifted ? ("F0自动校正×" + s.f0_octave_shifted) : "F0自动校正");
    if (s.backend && s.backend !== "basic-pitch") optTags.push("后端:" + s.backend);
    var optHtml = optTags.length
      ? '<div style="margin:0 0 14px">' + optTags.map(function (t) {
          return '<span class="badge">' + esc(t) + '</span>';
        }).join("") + '</div>'
      : "";

    box.innerHTML =
      '<div class="stats">' + statsHtml + '</div>' +
      optHtml +

      compareHtml +

      (f.staff ? (
        '<div id="staffBox"><h3>五线谱 <span class="tag">LilyPond</span></h3>' +
        (s.staff_truncated ?
          '<div class="msg err" style="margin-bottom:10px">⚠ 五线谱较长，仅显示前 ' + (s.staff_max_events || 600) + ' 个事件。完整音符请下载 <b>MIDI / MusicXML</b> 或查看<b>钢琴卷帘图</b>。</div>'
          : '') +
        '<img id="staff" class="score" alt="五线谱" /></div>'
      ) : '') +
      (f.pianoroll ? (
        '<h3>钢琴卷帘图 <span class="tag">Piano Roll</span></h3>' +
        '<img id="roll" class="score" alt="钢琴卷帘" />'
      ) : '') +
      '<h3>简谱 <span class="tag">标准简谱 · 1=' + (s.key || "C") + ' · 数字=唱名 上/下点=八度 下划线=八分·十六分 右横线=二分·全音符 0=休止</span>' +
        '<button id="copyJp" class="copyBtn">复制</button></h3>' +
      '<pre id="jianpu" class="jianpu"></pre>' +
      '<h3>下载</h3><div class="dl">' +
        (f.pianoroll ? '<a id="dlRoll" download>⬇ 钢琴卷帘（带响应强度 PNG）</a>' : '') +
        (f.midi ? '<a id="dlMidi" download>⬇ MIDI（可导入任意打谱软件）</a>' : '') +
        (f.musicxml ? '<a id="dlXml" download>⬇ MusicXML（带响应强度标注，可导入 MuseScore / Sibelius）</a>' : '') +
      '</div>' +
      (tip ? '<div class="hint">' + esc(tip.trim()) + '</div>' : '');

    if (f.audio) document.getElementById("audioOrig").src = f.audio;
    if (f.score_audio) document.getElementById("audioScore").src = f.score_audio;
    if (f.staff) {
      var st = document.getElementById("staff");
      st.src = f.staff;
      st.addEventListener("click", function (e) { window.open(e.target.src, "_blank"); });
    }
    if (f.pianoroll) {
      var rl = document.getElementById("roll");
      rl.src = f.pianoroll;
      rl.addEventListener("click", function (e) { window.open(e.target.src, "_blank"); });
    }
    document.getElementById("jianpu").textContent = rec.jianpu || "（无）";
    if (f.pianoroll) document.getElementById("dlRoll").href = f.pianoroll;
    if (f.midi) document.getElementById("dlMidi").href = f.midi;
    if (f.musicxml) document.getElementById("dlXml").href = f.musicxml;

    if (hasAudio) window.initPlayer(document.getElementById("audioOrig"),
      document.getElementById("playOrig"), document.getElementById("progOrig"),
      document.getElementById("timeOrig"));
    if (hasScore) window.initPlayer(document.getElementById("audioScore"),
      document.getElementById("playScore"), document.getElementById("progScore"),
      document.getElementById("timeScore"));

    // 对比播放：两者都有时，一个按钮同时播放/暂停两者
    if (hasAudio && hasScore) {
      var pb = document.getElementById("playBoth");
      var aO = document.getElementById("audioOrig"), aS = document.getElementById("audioScore");
      if (pb && aO && aS) {
        var labelBoth = "▶ 对比播放（原音频 + 钢琴谱）";
        pb.addEventListener("click", function () {
          if (aO.paused && aS.paused) { aO.play(); aS.play(); pb.textContent = "⏸ 暂停对比"; }
          else { aO.pause(); aS.pause(); pb.textContent = labelBoth; }
        });
        var stopBoth = function () { aO.pause(); aS.pause(); pb.textContent = labelBoth; };
        aO.addEventListener("pause", stopBoth);
        aS.addEventListener("pause", stopBoth);
        aO.addEventListener("ended", stopBoth);
        aS.addEventListener("ended", stopBoth);
      }
    }

    var cj = document.getElementById("copyJp");
    if (cj) cj.addEventListener("click", function () {
      navigator.clipboard.writeText(document.getElementById("jianpu").textContent).then(function () {
        var b = cj, t = b.textContent; b.textContent = "已复制 ✓";
        setTimeout(function () { b.textContent = t; }, 1500);
      });
    });

    box.classList.remove("hidden");
    window.scrollTo({ top: box.offsetTop - 20, behavior: "smooth" });
  };
})();
