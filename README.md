# DermAlert Framework 

DermAlert is an end-to-end computer vision and machine learning framework designed for smartphone-acquired dermatological imagery monitoring. The platform leverages advanced image pre-processing pipelines alongside non-linear decision classifiers to detect and evaluate tissue anomalies.

🔗 **Live Deployment URL:** [https://dermalert-analytics.onrender.com](https://dermalert-analytics.onrender.com)

---

##  Key Features

* **Advanced Pre-processing Pipeline:** Integrates adaptive contrast enhancement (CLAHE) and digital hair removal routines (Dull-Razor algorithm) to isolate pure lesion structures.
* **Intelligent Inference Engine:** Employs pre-trained machine learning classifiers to calculate statistical confidence metrics for targeted regions.
* **Edge-Case Triage Warning Matrix:** Features active mathematical variance tracking to flag structural noise (e.g., periorbital/facial boundary folds) on the SVM hyperplane.
* **Responsive Mobile Interface:** Clean, production-ready dashboard optimized for seamless smartphone and desktop utility.

---

## System Architecture & Workspace

* `app.py`: Main Flask application core handling routing and processing pipelines.
* `model.pkl`: Pre-trained mathematical classifier core.
* `templates/`: Interactive HTML UI workspace.
* `requirements.txt`: Environment dependencies profile.
* `render.yaml`: Automated blueprint layout configuration.

---

##  Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AayushiKumari-tech/DermAlert-Framework.git](https://github.com/AayushiKumari-tech/DermAlert-Framework.git)
   cd DermAlert-Framework
   ## Author
* Developed by **Aayushi Kumari** - B.Tech CSE (6th Sem)