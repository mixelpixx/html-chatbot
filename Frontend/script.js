function sendQuery() {
    let query = document.getElementById('query').value;
    fetch('http://localhost:5000/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({query: query}),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('response').innerText = JSON.stringify(data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
