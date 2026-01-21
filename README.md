<h1 align="center">Course Enrollment Guidance Chatbot 🤖</h1>
<h2 align="center"><i>Python Based Chatbot</i></h2>
<h3>Problem Statement</h3>
<p>📍<b><i>The Problem we aim to solve</i></b>
<ul>University students often face difficulty in selecting appropriate courses each semester due to:
o	Complex degree requirements
o	Multiple prerequisite dependencies
o	Limited availability of academic advisors
o	Lack of personalized academic guidance
As a result, students may enroll in unsuitable courses, causing:
o	Academic failure
o	Delayed graduation
o	Increased workload stress
This project proposes an AI-based Course Enrollment Guidance Chatbot that intelligently analyzes a student’s academic profile and recommends suitable courses and career-aligned guidance.</ul>
📍<b><i> Importance of Problem</i></b>
<ul><i>Academic relevance: Helps students make informed course enrollment decisions</i> </ul>
<ul><i>Institutional relevance: Reduces dependency on manual academic advisin </i></ul>
<ul><i>Societal relevance: Improves graduation rates and academic planning</i></ul>
<ul><i>Industrial relevance: Similar AI advisors are used in modern LMS and EdTech platforms</i></ul>
📍<b><i>The Scope and Assumptions</i></b>
<ul>The system is designed for BS Computer Science students</ul>
<ul>Student academic data is assumed to be accurate and complete</ul>
<ul>The chatbot provides recommendations, not mandatory decisions</ul>
<ul>The system does not replace human advisors, but assists them</ul>

<h3>Dataset Description</h3>
<b><i> •	Dataset Source</i></b>
<ul>o	Self – generated academic dataset</ul>
<b><i>•	Type of Dataset</i></b>
<ul>o	Tabular / Structured Data</ul>
<ul>o	Combination of textual and Categorical data</ul>
<h4>Dataset 1: AI Courses Dataset</h4>
<ul><b>Records: </b>60 courses</ul>
<ul><b>Key attributes: </b>
  <li>Course Name</li>
  <li>Course Level (Beginner / Intermediate / Advanced)</li>
  <li>Prerequisites</li>
  <li>Skills Covered</li>
  <li>Career Domain</li></ul>
<ul><b>Purpose</b>Course recommendation based on skill and level matching</ul>

<h4>Dataset 2: Career Mapping Dataset</h4>
<ul><b>Records: </b>12 career paths</ul>
<ul><b>Key attributes: </b>
  <li>Career Name</li>
  <li>Required Skills</li>
  <li>Recommended Courses</li></ul>
<ul><b>Purpose</b>Career-aligned course guidance</ul>

<h4>Dataset 3: Student Profiles Dataset</h4>
<ul><b>Records: </b>200 students</ul>
<ul><b>Key attributes: </b>
  <li>Completed courses</li>
  <li>Skills</li>
  <li>GPA</li>
  <li>Academic Interests</li></ul>
<ul><b>Purpose</b>Personalized recommendation generation</ul>

<h4>Dataset 4: Intent-Response Dataset</h4>
<ul><b>Records: </b>197 Intent-Response pairs</ul>
<ul><b>Key attributes: </b>
  <li>User Intent</li>
  <li>Chatbot Response</li></ul>
<ul><b>Purpose</b>Natural language interaction with students</ul>

<h3>Learning Model and Task Section</h3>
<b><i>•	Learning Tasks</i></b>
<ul>o	Classification: Identify student intent (career advice, course recommendation, prerequisite query)</ul>
<ul>o	Recommendation: Match student profiles with suitable courses</ul>
<ul>o	Similarity Matching: Compare student skills with course requirements</ul>
<ul>o	Input Processing: Speech-to-text conversion is applied before intent classification to allow voice-based student interaction.</ul>

<b><i>•	Selected Model</i></b>
<ul>o	TF-IDF Vectorizer + Logistic Regression: For intent classification</ul>
<ul>o	Cosine Similarity: For course and skill matching</ul>
<ul>o	Rule-assisted Decision Logic: For prerequisite validation</ul>

<b><i>•	Justification</i></b>
<ul>o	Logistic Regression is:
	Interpretable
	Efficient for text classification
	Suitable for small-to-medium datasets

</ul>
<ul>o	Cosine similarity works well for:
	Skill and course text comparison
</ul>
<ul>o	Hybrid AI approach ensures:
	Accuracy
	Explainability
	Academic feasibility</ul>

<h3>Learning Paradigm, Input & Output</h3>
<b><i>•	Learning Paradigm</i></b>
<ul>o	Supervised Learning
Intent classification using labeled intent-response data</ul>
<ul>o	Information Retrieval and Similarity Learning
Course Recommendation</ul>

<b><i>•	System Inputs</i></b>
<ul>o	Student profile (skills, GPA, completed courses)</ul>
<ul>o	User query (text input)</ul>
<ul>o	Career preference (optional)</ul>
<ul>o	The chatbot also supports speech-based interaction. Spoken queries are converted into text using a speech recognition module and processed through the same NLP and classification pipeline as text input.</ul>

<b><i>•	System Outpus</i></b>
<ul>o	Recommended Courses</ul>
<ul>o	Career Suggestions</ul>
<ul>o	Prerequisite warnings</ul>
<ul>o	Academic guidance messages</ul>

<h3>Performance Evaluation Metrics</h3>
<b><i>•	Metrics Used:</i></b>
<ul>o	Accuracy: Measures correctness of intent classification</ul>
<ul>o	Precision and Recall: Important to avoid wrong academic advice</ul>
<ul>o	F1 – Score: Balances precision and recall</ul>
<ul>o	Confusion Matrix: Analyzes misclassified intents</ul>

<b><i>•	Justification</i></b>
<ul>o	Since wrong recommendations can negatively impact students, precision and recall are critical</ul>
<ul>o	Accuracy alone is insufficient for educational systems</ul>

<b><i>•	System Outpus</i></b>
<ul>o	Recommended Courses</ul>
<ul>o	Career Suggestions</ul>
<ul>o	Prerequisite warnings</ul>
<ul>o	Academic guidance messages</ul>

<h3>PEAS Analysis</h3>
    <table>
        <tr>
            <th>Component</th>
            <th>Description</th>
        </tr>
        <tr>
            <td>Performance</td>
            <td>Accurate course and career recommendations</td>
        </tr>
        <tr>
            <td>Environment</td>
            <td>Partially observable, dynamic academic environment</td>
        </tr>
        <tr>
            <td>Actuators</td>
            <td>Display recommendations, responses, warnings</td>
        </tr>
        <tr>
            <td>Sensors</td>
            <td>Student input, profile data, course datasets</td>
        </tr>
    </table>

<b><i>•	Environment Characteristics</i></b>
<ul>o	Partially observable: Not all student preferences are known</ul>
<ul>o	Dynamic: Courses and skills evolve over time</ul>
<ul>o	Non-deterministic: Student decisions vary</ul>

<h3>Potential Applications</h3>
<b><i>•	Real World Applications</i></b>
<ul>o	University academic advising systems</ul>
<ul>o	Learning Management Systems (LMS)</ul>
<ul>o	EdTech platforms</ul>
<ul>o	Career counseling portals</ul>

<b><i>•	Users / Stakeholders</i></b>
<ul>o	University students</ul>
<ul>o	Academic advisors</ul>
<ul>o	Educational institutions</ul>
<ul>o	Curriculum designers</ul>

<b><i>•	Future Extensions</i></b>
<ul>o	Integration with real university databases</ul>
<ul>o	Use of deep learning models</ul>
<ul>o	GPA prediction</ul>
<ul>o	Multi-language chatbot</ul>
<ul>o	Mobile application deployment</ul>


<dl>
  <dt>Tool Used</dt>
  <dd> <a href="https://code.visualstudio.com/" target="_blank" rel="norefferer">  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSY2JEzCppRpKpOx5u62PHXEfEO2dCHNFUg2A&s" alt="VS code" width="40" height="40"></a></dd>
  <dt>Language Used</dt>
  <dd><a href="https://www.python.org/" target="_blank" rel="noreferrer" title="Java"> <img src="https://thebite.org/wp-content/uploads/2025/04/Python-1200x1200.webp" alt="java" width="40" height="40"/> </a></dd>
</dl></p>
