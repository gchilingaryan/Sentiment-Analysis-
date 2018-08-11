function classify() {
  var data = {"text": document.getElementById("tweetText").value}

  $.post("/classify", data, function(data, status){
    var img = document.getElementById('sentimentImg')
    if (data.sentiment == 'Positive') {
      img.setAttribute('src', 'static/img/happy.svg');
    } else {
      img.setAttribute('src', 'static/img/sad.svg');
    }
    document.getElementById('probText').textContent = '(' + data.sentiment + ' with probability: ' + data.probability + ')'
  }, "json");
}

$("#tweet_form").submit(function(e) {
    e.preventDefault();
});