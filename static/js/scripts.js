var socket = io.connect('http://' + document.domain + ':' + location.port + '/update');

socket.on('progress_update', function(data) {
    // Update the progress in the table based on the received data
    var experimentId = 'experiment_' + data.starting_time;
    var progressCell = document.getElementById(experimentId);
    if (progressCell) {
        progressCell.innerHTML = data.progress;
    }
});

socket.on('progress_complete', function(data) {
    // trigger deletion of a row in running jobs and adding a new row in result
    var experimentId = data.starting_time;
    var experimentParams = data.params;
    var experimentBestParams = data.best_params;
    var experimentCVAcc = data.cv_acc;
    var experimentAccuracy = data.accuracy;

    // delete finished experiment
    var running_exp = document.getElementById("running_experiments");
    for (var i = 0; i < running_exp.rows.length; i++) {
        if (running_exp.rows[i].id === experimentId) {
            running_exp.deleteRow(i);
            break;
        }
    }

    // adding new rows to the finished experiments
    var finished_exp = document.getElementById("finished_experiments");
    var rowCount = finished_exp.tBodies[0].rows.length;
    if (rowCount == 1) {
        var new_exp_pos = 1;
    } else {
        var new_exp_pos = 1;
        // find the position for the finished experiment
        for (var i = 1; i < rowCount; i++) {
            console.log(finished_exp.rows[i].cells[8].innerText.slice(0, -1));
            if (finished_exp.rows[i].cells[8].innerText.slice(0, -1) >= experimentAccuracy) {
                new_exp_pos += 1;
            } else {
                break;
            }
        }
    }
    var newRow = finished_exp.insertRow(new_exp_pos);
    var hld = newRow.insertCell(0);
    var dr = newRow.insertCell(1);
    var op = newRow.insertCell(2);
    var lr = newRow.insertCell(3);
    var bs = newRow.insertCell(4);
    var epoch = newRow.insertCell(5);
    var bp = newRow.insertCell(6);
    var cv_acc = newRow.insertCell(7);
    var acc = newRow.insertCell(8);
    hld.innerHTML = "[" + experimentParams.hidden_layers_dim + "]";
    if (experimentParams.dropout_rate == "None") {
        dr.innerHTML = experimentParams.dropout_rate;
    } else if (experimentParams.dropout_rate.length == 1) {
        dr.innerHTML = experimentParams.dropout_rate[0];
    } else {
        dr.innerHTML = "[" + experimentParams.dropout_rate + "]";
    }
    op.innerHTML =  experimentParams.optimizer;
    lr.innerHTML = "[" + experimentParams.learning_rate + "]";
    bs.innerHTML = "[" + experimentParams.batch_size + "]";
    epoch.innerHTML = "[" + experimentParams.num_epochs + "]";
    bp.innerHTML = "[" + Object.values(experimentBestParams) + "]";
    if (experimentCVAcc == "None") {
        cv_acc.innerHTML = experimentCVAcc;
    } else {
        cv_acc.innerHTML = experimentCVAcc + "%";
    }
    acc.innerHTML = experimentAccuracy + "%";
});

function toggleDropdown() {
    var userGuide = document.getElementById("userGuide");
    var userGuideHeader = document.getElementById("userGuideHeader");
    if (userGuide.style.display === "none") {
        userGuide.style.display = "block";
        userGuideHeader.innerText = "How to use this app: "
    } else {
        userGuide.style.display = "none";
        userGuideHeader.innerText = "Click here to for how to use: "
    }
}