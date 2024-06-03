if ('webkitSpeechRecognition' in window) 
{
  console.log("speech recognition API supported");
} 
else 
{
  console.log("speech recognition API not supported")
}

  
var recognition = new webkitSpeechRecognition();
var listen = false;

recognition.continuous = true;
var Textbox = document.getElementById("query-box");

var instructions = document.getElementById("instructions");
console.log(instructions);
console.log(Textbox);
var Content = '';

recognition.continuous = true;

recognition.onresult = function(event) {

  var current = event.resultIndex;

  var transcript = event.results[current][0].transcript;
 
    Content += transcript;
    console.log(Content);
    Textbox.value = Content;
    
  
};

var button = document.getElementById("start-btn");

recognition.onstart = function() { 
  console.log("On Start");
  listen = true;
  button.textContent = 'Voice recognition is ON.';
}

recognition.onstop = function() {
  console.log("On Stop");
  listen = false;
  button.textContent = 'Start Voice Query';
}

recognition.onerror = function(event) {
  console.log("Error");
  listen = false;
  console.error(event);
  if(event.error == 'no-speech') {
    instructions.text('Try again.');  
  }
}

document.getElementById("start-btn").onclick = function(e) {
  if(!listen)
  {
    if (Content.length) {
      Content += ' ';
    }
    console.log("Starting");
    recognition.start();
  }
  else
  {
    listen = false;
    console.log("Stopping");
    recognition.stop()
  }
};

Textbox.oninput = function() {
  Content = $(this).val();
}