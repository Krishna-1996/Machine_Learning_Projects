document.getElementById('mazeForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    
    const algorithm = document.getElementById('algorithm').value;
    const obstacle = document.getElementById('obstacle').value;
    const goal = document.getElementById('goal').value;

    const response = await fetch('/solve_maze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithm, obstacle_percentage: obstacle, goal_position: goal })
    });

    const data = await response.json();
    
    if (data.error) {
        alert("Error: " + data.error);
    } else {
        drawMaze(data.maze);
    }
});

function drawMaze(mazeData) {
    const canvas = document.getElementById('mazeCanvas');
    const ctx = canvas.getContext('2d');

    const rows = mazeData.length;
    const cols = mazeData[0].length;
    const cellSize = 10;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            ctx.fillStyle = mazeData[row][col] === 1 ? 'black' : 'white';
            ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
            ctx.strokeRect(col * cellSize, row * cellSize, cellSize, cellSize);
        }
    }
}
