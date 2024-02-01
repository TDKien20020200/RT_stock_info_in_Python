function doSomethingWithTicker(ticker) {
    console.log("Ticker:", ticker);
}

function showHistoryData(ticker) {
    $('#history-data').append(`
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Close Price</th>
                </tr>
            </thead>
            <tbody>
                {% for item in data_list %}
                    <tr>
                        <td>{{ item['Date'] }}</td>
                        <td>{{ item['Close'] }}</td>
                    </tr>
                {% endfor %}
            </tbody>
        </table>
    `);
}