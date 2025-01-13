Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Drop your image here",
        autoProcessQueue: false,
        dictRemoveFile: "",
        dictResponseError: ""
    });
    
    dz.on("addedfile", function() {
        if (dz.files[1]!=null) {
            dz.removeFile(dz.files[0]);        
        }
    });

    dz.on("complete", function (file) {
        let imageData = file.dataURL;
        
        var url = "http://127.0.0.1:5000/classify_emotion";

        $.post(url, {
            image_data: file.dataURL
        },function(data, status) {
            if (!data || data.length==0) {
                $("#resultHolder").hide();
                $("#divClassTable").hide();                
                $("#error").show();
                return;
            }
            let emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise"];
            
            let match = null;
            let bestScore = -1;
            for (let i=0;i<data.length;++i) {
                let maxScoreForThisClass = Math.max(...data[i].class_probability);
                if(maxScoreForThisClass>bestScore) {
                    match = data[i];
                    bestScore = maxScoreForThisClass;
                }
            }
            if (match) {
                $("#error").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();
                $("#resultHolder").html($(`[data-emotion="${match.class}"`).html());
                let classDictionary = match.class_dictionary;
                for(let emotion in classDictionary) {
                    let index = classDictionary[emotion];
                    let proabilityScore = match.class_probability[index];
                    let emotionName = "#score_" + emotion;
                    $(emotionName).html(proabilityScore);
                }
            }           
        });
    });

    $("#submitBtn").on('click', function (e) {
        dz.processQueue();		
    });
}

$(document).ready(function() {
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();

    init();
});