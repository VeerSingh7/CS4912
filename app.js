// Import Firebase SDKs
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { getFirestore, doc, getDoc, setDoc, onSnapshot, updateDoc, increment, serverTimestamp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore.js";

// --- CONFIGURATION ---
// TODO: Replace with your actual Firebase project config
const firebaseConfig = {
    apiKey: "AIzaSy_YOUR_API_KEY_HERE",
    authDomain: "your-project.firebaseapp.com",
    projectId: "your-project-id",
    storageBucket: "your-project.appspot.com",
    messagingSenderId: "123456789",
    appId: "1:123456789:web:abcdef"
};

// Check if using placeholder config
const isMockMode = firebaseConfig.projectId === "your-project-id";

// Initialize Firebase
let app, db;

if (!isMockMode) {
    try {
        app = initializeApp(firebaseConfig);
        db = getFirestore(app);
        console.log("Firebase initialized");
    } catch (e) {
        console.warn("Firebase initialization failed.", e);
    }
} else {
    console.warn("Using MOCK MODE (LocalStorage) because Firebase config is missing. Votes will be saved locally only.");
}

// Global User State (Anonymous ID)
let currentUserId = localStorage.getItem("cv_visitor_id");
if (!currentUserId) {
    currentUserId = 'anon_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem("cv_visitor_id", currentUserId);
}
console.log("User ID:", currentUserId);


// --- VOTING LOGIC ---

// Expose vote function globally
window.vote = async function (taskId, value) {

    // MOCK MODE HANDLER
    if (isMockMode) {
        handleMockVote(taskId, value);
        return;
    }

    // REAL FIREBASE HANDLER
    if (!db) {
        alert("Database not connected. Check console.");
        return;
    }

    const voteId = `${currentUserId}_${taskId}`;
    const taskRef = doc(db, "tasks", taskId);
    const voteRef = doc(db, "votes", voteId);

    try {
        const voteSnap = await getDoc(voteRef);

        let previousValue = 0;
        if (voteSnap.exists()) {
            previousValue = voteSnap.data().voteValue;
        }

        // Write the vote
        await setDoc(voteRef, {
            userId: currentUserId,
            voteValue: value,
            taskId: taskId,
            timestamp: serverTimestamp()
        });

        // Update aggregate score
        const diff = value - previousValue;
        if (diff !== 0) {
            // Create task doc if not exists
            await setDoc(taskRef, { id: taskId }, { merge: true });
            await updateDoc(taskRef, {
                score: increment(diff)
            });
        }

    } catch (e) {
        console.error("Voting failed", e);
    }
};


// --- REALTIME LISTENERS ---

const tasks = ['exp1', 'exp2', 'exp3'];

// Firebase Listeners
if (!isMockMode && db) {
    tasks.forEach(taskId => {
        onSnapshot(doc(db, "tasks", taskId), (doc) => {
            updateCountUI(taskId, doc.exists() ? doc.data().score : 0);
        });
    });
} else {
    // Initial Mock Load
    tasks.forEach(taskId => {
        const mockScore = parseInt(localStorage.getItem(`mock_score_${taskId}`) || "0");
        updateCountUI(taskId, mockScore);
    });
}

// UI Helper
function updateCountUI(taskId, score) {
    const el = document.getElementById(`count-${taskId}`);
    if (el) {
        el.textContent = score || 0;
        if ((score || 0) > 0) el.style.color = "var(--success)";
        else if ((score || 0) < 0) el.style.color = "var(--error)";
        else el.style.color = "var(--text-primary)";
    }
}

// --- MOCK IMPLEMENTATION ---
function handleMockVote(taskId, value) {
    // 1. Get user's previous vote for this task
    const userVoteKey = `mock_vote_${currentUserId}_${taskId}`;
    const prevVote = parseInt(localStorage.getItem(userVoteKey) || "0");

    // 2. Update user's vote
    localStorage.setItem(userVoteKey, value);

    // 3. Update total score
    const taskScoreKey = `mock_score_${taskId}`;
    let currentScore = parseInt(localStorage.getItem(taskScoreKey) || "0");

    const diff = value - prevVote;
    currentScore += diff;

    localStorage.setItem(taskScoreKey, currentScore);

    // 4. Update UI
    updateCountUI(taskId, currentScore);

    console.log(`[Mock] Voted ${value} for ${taskId}. New Score: ${currentScore}`);
}
