$(document).ready(function () {

    $("#div-media").show()

    // Form Submit and Show Overlay.
    $("#btn-find-sentiment").click((e) => {
        e.preventDefault();

        let formData = new FormData();

    if ($("#div-media").is(":visible")) {
        formData.append("input", $("#form-input-media")[0].files[0]);
        formData.append("type", "media")
    }

        if (formData.get("input") != '') {
            $("#sentiment-overlay").css("display", "block");
            $.ajax({
                type: "POST",
                url: '/',
                data: formData,
                dataType: 'json',
                contentType: false,
                processData: false,
            })
                .done((response) => {
                    val_neu = Math.round(parseFloat(response["score_neutral"]) * 10000) / 100
                    val_cal = Math.round(parseFloat(response["score_calm"]) * 10000) / 100
                    val_hap = Math.round(parseFloat(response["score_happy"]) * 10000) / 100
                    val_sad = Math.round(parseFloat(response["score_sad"]) * 10000) / 100
                    val_ang = Math.round(parseFloat(response["score_angry"]) * 10000) / 100
                    val_fea = Math.round(parseFloat(response["score_fearful"]) * 10000) / 100
                    val_dis = Math.round(parseFloat(response["score_disgust"]) * 10000) / 100
                    val_sur = Math.round(parseFloat(response["score_surprised"]) * 10000) / 100
                    
                    $("#sentiment-overlay-content").html(`
                        <h2>PROMINENT SENTIMENT</h2>
                        <p id="prominent-sentiment">` + response["prominent_sentiment"] + `</p>
                        <div id="sentiment-info">
                            <div id="sentiment-table">
                                <table class="table">
                                    <thead>
                                        <th scope="col">&#128566;</th>
                                        <th scope="col">SENTIMENT</th>
                                        <th scope="col">VALUE</th>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>&#128528;</td>
                                            <td>neutral</td>
                                            <td>` + val_neu + `%</td>
                                        </tr>
                                        <tr>
                                            <td>&#128524;</td>
                                            <td>calm</td>
                                            <td>` + val_cal + `%</td>
                                        </tr>
                                        <tr>
                                            <td>&#128512;</td>
                                            <td>happy</td>
                                            <td>` + val_hap + `%</td>
                                        </tr>
                                        <tr>
                                            <td>&#128532;</td>
                                            <td>sad</td>
                                            <td>` + val_sad + `%</td>
                                        </tr>
                                        <tr>
                                            <td>&#128544;</td>
                                            <td>angry</td>
                                            <td>` + val_ang + `%</td>
                                        </tr>
                                        <tr>
                                            <td>&#128552;</td>
                                            <td>fearful</td>
                                            <td>` + val_fea + `%</td>
                                        </tr>
                                        <tr>
                                            <td>&#129314;</td>
                                            <td>disgust</td>
                                            <td>` + val_dis + `%</td>
                                        </tr>
                                        <tr>
                                            <td>&#128562;</td>
                                            <td>surprised</td>
                                            <td>` + val_sur + `%</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                            <canvas id="sentiment-radar-chart"></canvas>
                        </div>
                    `)

                    const chart = $("#sentiment-radar-chart")
                    new Chart(chart, {
                        type: 'radar',
                        data: {
                            labels: ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
                            datasets: [{
                                label: 'Sentiment',
                                data: [val_neu, val_cal, val_hap, val_sad,val_ang,val_fea,val_dis,val_sur],
                                fill: true,
                                color: '#FFFFFF',
                                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                                borderColor: 'rgb(255, 99, 132)',
                                pointBackgroundColor: '#FFFFFF',
                                pointBorderColor: '#FFFFFF',
                                pointHoverBackgroundColor: '#FFFFFF',
                                pointHoverBorderColor: '#FFFFFF'
                            }]
                        },
                        options: {
                            elements: {
                                line: {
                                    borderWidth: 5
                                },
                                legend: {
                                    labels: {
                                        color: 'rgb(255, 255, 255)',
                                        fontSize: 18
                                    }
                                }
                            },
                            scales: {
                                r: {
                                    angleLines: {
                                        color: 'rgba(255, 255, 255, 0.8)'
                                    },
                                    grid: {
                                        color: 'rgba(255, 255, 255, 0.8)'
                                    },
                                    pointLabels: {
                                        color: 'rgb(255, 255, 255)',
                                        fontSize: 20
                                    },
                                    ticks: {
                                        color: 'rgb(0, 0, 0)'
                                    }
                                }
                            }
                        }
                    });
                })
                .fail((err) => {
                    console.log(err)
                })
        }

    });

    // Remove overlay
    $("#sentiment-overlay").click(() => {
        $("#sentiment-overlay").css("display", "none");
        $("#form-input-textarea").val("");
        $("#form-input-url").val("");
        $("#form-input-media").val("");
        window.location.reload()
    });

});