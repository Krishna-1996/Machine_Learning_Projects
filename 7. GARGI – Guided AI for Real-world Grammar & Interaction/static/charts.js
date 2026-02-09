const ctx = document.getElementById("scoreChart");

new Chart(ctx, {
  type: "line",
  data: {
    labels: scores.map((_, i) => i + 1),
    datasets: [{
      label: "Score",
      data: scores
    }]
  }
});
