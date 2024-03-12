var ticker = '{{ ticker }}';
        var historyDataShown = false;
        var overviewChartShown = false;
        var dailyChartShown = false;
        var weeklyChartShown = false;
        var maChartShown = false;
        var percentageChangeShown = false;
        var predictVisualizeShown = false;
        var predictChartShown = false;
        var predDataShown = false;

        $('#show_history_data_button').click(function() {
            if (historyDataShown) {
                $('#history-data').hide();
                historyDataShown = false;
            } else {
                $('#history-data').show();
                historyDataShown = true;
            }
        });

        $('#show_chart_button').click(function() {
            if (overviewChartShown) {
                $('#overview-chart-container').hide();
                overviewChartShown = false;
            } else {
                $('#overview-chart-container').show();
                showChart();
                overviewChartShown = true;
            }
        });

        $('#show_daily_chart_button').click(function() {
            if (dailyChartShown) {
                $('#daily-chart-container').hide();
                dailyChartShown = false;
            } else {
                $('#daily-chart-container').show();
                openImage('{{ url_for('static', filename='image/daily_chart.png') }}', "daily-chart-image", "daily-chart-container")
                dailyChartShown = true;
            }
        });

        $('#show_weekly_chart_button').click(function() {
            if (weeklyChartShown) {
                $('#weekly-chart-container').hide();
                weeklyChartShown = false;
            } else {
                $('#weekly-chart-container').show();
                openImage('{{ url_for('static', filename='image/weekly_chart.png') }}', "weekly-chart-image", "weekly-chart-container")
                weeklyChartShown = true;
            }
        });

        $('#show_ma_chart_button').click(function() {
            if (maChartShown) {
                $('#ma-chart-container').hide();
                maChartShown = false;
            } else {
                $('#ma-chart-container').show();
                openImage('{{ url_for('static', filename='image/MAs_chart.png') }}', "ma-chart-image", "ma-chart-container")
                maChartShown = true;
            }
        });

        $('#show_percentage_change_button').click(function() {
            if (percentageChangeShown) {
                $('#percentage-change-container').hide();
                percentageChangeShown = false;
            } else {
                $('#percentage-change-container').show();
                openImage('{{ url_for('static', filename='image/daily_return_chart.png') }}', "percentage-change-image", "percentage-change-container")
                percentageChangeShown = true;
            }
        });

        $('#show_prediction_button').click(function() {
            if (predictVisualizeShown) {
                $('#predict-visualize-container').hide();
                predictVisualizeShown = false;
            } else {
                $('#predict-visualize-container').show();
                openImage('{{ url_for('static', filename='image/prediction_chart.png') }}', "predict-visualize-image", "predict-visualize-container")
                predictVisualizeShown = true;
            }
        });

        $('#candlestick_prediction_chart_button').click(function() {
            if (predictChartShown) {
                $('#predict-candlestick-container').hide();
                predictChartShown = false;
            } else {
                $('#predict-candlestick-container').show();
                showPredictedChart();
                predictChartShown = true;
            }
        });

        $('#show_next_10_days_button').click(function() {
            if (predDataShown) {
                $('#pred-data').hide();
                predDataShown = false;
            } else {
                $('#pred-data').show();
                predDataShown = true;
            }
        });

        function openImage(url, imgId, divId) {
          document.getElementById(imgId).src = url;
          document.getElementById(divId).style.display = "block";
        }

        function showChart() {
            var candlestickTrace = {
                x: {{ chart_data['x'] | tojson }},
                open: {{ chart_data['open'] | tojson }},
                high: {{ chart_data['high'] | tojson }},
                low: {{ chart_data['low'] | tojson }},
                close: {{ chart_data['close'] | tojson }},
                name: 'Candlestick',
                type: 'candlestick'
            };
            var layout = {
                title: 'Candlestick Chart',
                xaxis: {
                    title: 'Date'
                },
                yaxis: {
                    title: 'Price',
                    rangemode: 'normal'
                }
            };
            var data = [candlestickTrace];
            Plotly.newPlot('overview-chart', data, layout);
        }

        function showPredictedChart() {
            var candlestickTrace = {
                x: {{ data_chart_predicted['x'] | tojson }},
                open: {{ data_chart_predicted['open'] | tojson }},
                high: {{ data_chart_predicted['high'] | tojson }},
                low: {{ data_chart_predicted['low'] | tojson }},
                close: {{ data_chart_predicted['close'] | tojson }},
                name: 'Candlestick',
                type: 'candlestick'
            };
            var layout = {
                title: 'Candlestick predicted Chart',
                xaxis: {
                    title: 'Date'
                },
                yaxis: {
                    title: 'Price',
                    rangemode: 'normal'
                }
            };
            var data = [candlestickTrace];
            Plotly.newPlot('predict-chart', data, layout);
        }