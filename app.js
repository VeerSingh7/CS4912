// Simple Local Storage Voting System

// Configuration
const TASKS = ['exp1', 'exp2', 'exp3'];

// Initialize UI on load
document.addEventListener('DOMContentLoaded', () => {
    TASKS.forEach(taskId => {
        updateUI(taskId);
        checkIfVoted(taskId);
    });
});

// Expose vote function globally
window.vote = function (taskId, value) {
    // 1. Check if already voted
    if (localStorage.getItem(`voted_${taskId}`)) {
        alert("You have already voted on this task!");
        return;
    }

    // 2. Update Count
    const countKey = `count_${taskId}`;
    let currentCount = parseInt(localStorage.getItem(countKey) || "0");
    currentCount += value;
    localStorage.setItem(countKey, currentCount);

    // 3. Mark as voted to prevent multiple votes
    localStorage.setItem(`voted_${taskId}`, "true");

    // 4. Update UI
    updateUI(taskId);
    checkIfVoted(taskId);
};

function updateUI(taskId) {
    const el = document.getElementById(`count-${taskId}`);
    if (el) {
        const count = parseInt(localStorage.getItem(`count_${taskId}`) || "0");
        el.textContent = count;

        // Color coding
        if (count > 0) el.style.color = "#00cc66"; // Success color
        else if (count < 0) el.style.color = "#ff4d4d"; // Error color
        else el.style.color = "inherit";
    }
}

function checkIfVoted(taskId) {
    if (localStorage.getItem(`voted_${taskId}`)) {
        // Disable buttons for this task
        const section = document.getElementById(`vote-${taskId}`);
        if (section) {
            const btns = section.querySelectorAll('.vote-btn');
            btns.forEach(btn => {
                btn.disabled = true;
                btn.style.opacity = "0.5";
                btn.style.cursor = "not-allowed";
            });
        }
    }
}
