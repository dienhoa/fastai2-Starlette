var el = x => document.getElementById(x);

var loadAudio = function(event) {
  var path = event.target.files[0].name;
  el('upload-label').innerHTML = path;
  var reader = new FileReader();
  reader.readAsDataURL(event.target.files[0])
  reader.onload = function(e) {
    el('audio-picked').src = e.target.result;
    el('audio-picked').className = ""; 
    el('analyze-button').className = ""; 
  }
}

function analyze() {
  var uploadFiles = el("input-audio").files;
  if (uploadFiles.length !== 1) alert("Please select a file to analyze!");

  el("result-label").innerHTML = `Not finish yet`;
  el("analyze-button").innerHTML = "Analyzing...";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,
    true);
    
  xhr.onerror = function() {
    alert(xhr.responseText);
  };

  xhr.onreadystatechange = function(e) {
    if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
      var response = JSON.parse(e.target.responseText);
      el("result-label").innerHTML = `Result = ${response["result"]}`;
    }
    el("analyze-button").innerHTML = "Analyze";
  };

  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}

