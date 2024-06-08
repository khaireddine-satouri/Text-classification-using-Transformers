document.getElementById('classify-form').addEventListener('submit', async function(e) {
    e.preventDefault();

    const text = document.getElementById('text-input').value;
    const response = await fetch('/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
    });

    const result = await response.json();
    document.getElementById('result').innerText = `Prediction: ${result.prediction}, Score: ${result.score}`;
});
