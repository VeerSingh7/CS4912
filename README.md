# Computer Vision Portfolio & Voting System

## Setup Instructions

### 1. Firebase Configuration (Required for Voting)

To make the voting system work, you need your own Firebase project.

1.  Go to [Firebase Console](https://console.firebase.google.com/).
2.  Create a new project.
3.  **Authentication**: Enable **Google Sign-In** in the *Authentication -> Sign-in method* tab.
4.  **Firestore Database**: Create a database (start in **Test Mode** for easier development).
5.  **Project Settings**:
    *   Go to Project Settings (gear icon).
    *   Scroll down to "Your apps" and click the `</>` (Web) icon.
    *   Register the app (e.g., "CV-Portfolio").
    *   Copy the `firebaseConfig` object (apiKey, authDomain, etc.).
6.  **Code Update**:
    *   Open `app.js`.
    *   Replace the `firebaseConfig` placeholder object with your actual keys.

### 2. Deployment to GitHub Pages

1.  Push this code to your GitHub repository.
2.  Go to the Repository **Settings**.
3.  Go to the **Pages** section on the left sidebar.
4.  Under **Build and deployment** / **Source**, select `Deploy from a branch`.
5.  Select `main` (or `master`) branch and `/ (root)` folder.
6.  Click **Save**.
7.  Wait a minute, and your site will be live at `https://[username].github.io/cv` (or root if repo is named username.github.io).

### 3. Usage

*   **Images**: The current images are AI-generated placeholders. You should replace them in `assets/images/` with your actual photography experiment results.
*   **Voting**: Users must sign in with Google to vote. The application limits one vote per student per task.

## Directory Structure

*   `index.html`: Main entry point.
*   `style.css`: All styling (Dark mode, glassmorphism).
*   `app.js`: Connects to Firebase and handles logic.
*   `assets/images/`: Stores your photos.
