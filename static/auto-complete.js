  function GetAutoComplete(e)
  {
    console.log("Entered GetAutoComplete")
    var query = e.value;
    
    var auto = [];

    var uri = 'http://localhost:5000/auto-complete?q='+query;
    uri = encodeURI(uri);

    fetch(uri)
    .then(resp => resp.json())
    .then(function(resp){
        console.log(resp)
        resp.auto_complete.forEach(element => {
            auto.push(element);
        });

        console.log(auto);
        $( "#query-box" ).autocomplete({
            source: auto
          });
    });
  }