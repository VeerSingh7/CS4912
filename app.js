// Import Firebase SDKs
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { getAuth, signInWithPopup, GoogleAuthProvider, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";
import { getFirestore, doc, getDoc, setDoc, onSnapshot, updateDoc, increment, serverTimestamp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore.js";

// --- CONFIGURATION ---
// TODO: Replace with your actual Firebase project config
// You can get this from the Firebase Console -> Project Settings -> General
const firebaseConfig = {
  apiKey: "AIzaSy_YOUR_API_KEY_HERE",
  authDomain: "your-project.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-project.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abcdef"
};

// Initialize Firebase
// Note: We wrap this in a try/catch so the site doesn't crash if config is invalid during development
let app, auth, db;
try {
    app = initializeApp(firebaseConfig);
    auth = getAuth(app);
    db = getFirestore(app);
    console.log("Firebase initialized");
} catch (e) {
    console.warn("Firebase initialization failed (probably missing config). Voting will be disabled.", e);
}

// Global User State
let currentUser = null;

// DOM Elements
const authBtn = document.getElementById('auth-btn');
const voteButtons = document.querySelectorAll('.vote-btn');

// --- AUTHENTICATION ---

const provider = new GoogleAuthProvider();
// Optional: restrict to domain
// provider.setCustomParameters({ hd: "columbia.edu" }); 

function handleAuth() {
    if (!auth) return alert("Firebase not configured!");

    if (currentUser) {
        signOut(auth).then(() => {
            console.log("Signed out");
        });
    } else {
        signInWithPopup(auth, provider)
            .then((result) => {
                // User signed in
                console.log("Signed in as", result.user.email);
            }).catch((error) => {
                console.error("Auth Error", error);
                alert("Login Failed: " + error.message);
            });
    }
}

if (authBtn) {
    authBtn.addEventListener('click', handleAuth);
}

if (auth) {
    onAuthStateChanged(auth, (user) => {
        currentUser = user;
        if (user) {
            authBtn.textContent = "Sign Out (" + user.email.split('@')[0] + ")";
            enableVoting(true);
        } else {
            authBtn.textContent = "Sign In with Google";
            enableVoting(false);
        }
    });
} else {
    // If no firebase, disable all buttons
    enableVoting(false);
    authBtn.textContent = "Auth Config Missing";
}

function enableVoting(enabled) {
    voteButtons.forEach(btn => {
        btn.disabled = !enabled;
        if (!enabled) {
            btn.title = "Please sign in to vote";
        } else {
            btn.title = "";
        }
    });
}


// --- VOTING LOGIC ---

// Expose vote function globally for onclick handlers
window.vote = async function(taskId, value) {
    if (!currentUser) {
        alert("Please sign in to vote!");
        return;
    }
    if (!db) return;

    const studentId = currentUser.email.replace(/[@.]/g, '_'); // sanitize email for ID
    const voteId = `${studentId}_${taskId}`;
    const taskRef = doc(db, "tasks", taskId);
    const voteRef = doc(db, "votes", voteId);

    // 1. Check if user already voted on this task
    // We will use a transaction or simple read/write.
    // Ideally user should only have 1 vote document per task.
    
    // Optimistic UI update could happen here, but better to wait for DB listener
    
    try {
        const voteSnap = await getDoc(voteRef);
        
        let previousValue = 0;
        if (voteSnap.exists()) {
            previousValue = voteSnap.data().voteValue;
        }

        // If clicking the same vote again, maybe toggle off? Or just update.
        // Rule: "Check if document exists... update vote"
        
        // Write the vote
        await setDoc(voteRef, {
            userId: currentUser.uid,
            email: currentUser.email,
            voteValue: value,
            taskId: taskId,
            timestamp: serverTimestamp()
        });

        // Update the aggregate count on the Task document (optional but good for performance)
        // Or just let the client calculate from raw votes (bad for scale, good for tiny demo)
        // Let's assume there is a document `tasks/exp1` that holds a `totalVotes` field.
        // To keep it simple for this assignment, we will use a separate listener to count votes or 
        // just increment a counter field.
        
        // Simple Counter Increment approach
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
        alert("Error saving vote: " + e.message);
    }
};


// --- REALTIME LISTENERS ---

// Listener for scores
const tasks = ['exp1', 'exp2', 'exp3'];

tasks.forEach(taskId => {
    if (!db) return;
    
    // Listen to the aggregated score
    onSnapshot(doc(db, "tasks", taskId), (doc) => {
        const el = document.getElementById(`count-${taskId}`);
        if (doc.exists() && el) {
            const data = doc.data();
            el.textContent = data.score || 0;
            
            // Color logic
            if ((data.score || 0) > 0) el.style.color = "var(--success)";
            else if ((data.score || 0) < 0) el.style.color = "var(--error)";
            else el.style.color = "var(--text-primary)";
        }
    });
});
