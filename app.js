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

// Initialize Firebase
let app, db;
try {
    app = initializeApp(firebaseConfig);
    db = getFirestore(app);
    console.log("Firebase initialized");
} catch (e) {
    console.warn("Firebase initialization failed. Voting will be disabled.", e);
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
        // alert("Error saving vote: " + e.message);
    }
};


// --- REALTIME LISTENERS ---

const tasks = ['exp1', 'exp2', 'exp3'];

tasks.forEach(taskId => {
    if (!db) return;

    onSnapshot(doc(db, "tasks", taskId), (doc) => {
        const el = document.getElementById(`count-${taskId}`);
        if (doc.exists() && el) {
            const data = doc.data();
            el.textContent = data.score || 0;

            if ((data.score || 0) > 0) el.style.color = "var(--success)";
            else if ((data.score || 0) < 0) el.style.color = "var(--error)";
            else el.style.color = "var(--text-primary)";
        }
    });
});
