document.addEventListener("DOMContentLoaded", function() {

  if (window.navigator.userAgent.indexOf("Mac") != -1) {
    document.getElementById("sb-kbd").innerHTML = "Cmd"; 
  } else {
    document.getElementById("sb-kbd").innerHTML = "Ctrl"; 
  }
  
});

function removeKBD() {
  document.getElementById("sb-kbd-span").style.display = "none";
}

function addKBD() {
  document.getElementById("sb-kbd-span").style.display = "block";
}
