import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

training_data = [
    # ============================================================
    # JOB_OKAY — Legitimate campus service requests
    # ============================================================
    ("I need a graphic designer to create a flyer for my event by Friday. Budget: ₦3,000", "job_okay"),
    ("Need a flyer designer for campus event budget 5k", "job_okay"),
    ("Looking for a calculus tutor for GST101 prep this evening", "job_okay"),
    ("GST tutor needed urgently before exam", "job_okay"),
    ("Need someone to run to the campus gate and pick up a package", "job_okay"),
    ("Errand runner needed from main gate to hostel B", "job_okay"),
    ("Photography needed for departmental dinner on Saturday", "job_okay"),
    ("Video editor needed to edit a 2-minute reels video", "job_okay"),
    ("Need someone to proofread my final year project document", "job_okay"),
    ("Assignment formatting and typing help needed", "job_okay"),
    ("Looking for a logo designer for my student business", "job_okay"),
    ("Need a web developer to build a simple portfolio website budget 10k", "job_okay"),
    ("Typist needed to type my handwritten lecture notes 2k per chapter", "job_okay"),
    ("Need a hair braider available on campus Saturday morning", "job_okay"),
    ("Looking for someone to help me with my physics practical report", "job_okay"),
    ("Data entry job 3 hours of work budget 4000 naira", "job_okay"),
    ("Need a makeup artist for graduation photos next week", "job_okay"),
    ("Looking for someone to teach me Python basics two sessions", "job_okay"),
    ("Need a voiceover artist for a 1 minute video script", "job_okay"),
    ("Social media manager needed to run my Instagram page for a month", "job_okay"),
    ("I need a translator to translate a document from English to Yoruba", "job_okay"),
    ("Looking for someone to transcribe a 10 minute audio recording", "job_okay"),
    ("Need a beat maker to produce a simple instrumental for a project", "job_okay"),
    ("Laundry service needed pickup from hostel A deliver same day", "job_okay"),
    ("Looking for someone to submit my departmental form at the admin block", "job_okay"),
    ("Proofreader needed for my 5000 word essay submission deadline tomorrow", "job_okay"),
    ("Need someone to design a PowerPoint presentation 10 slides", "job_okay"),
    ("Looking for a CV designer to help format my resume professionally", "job_okay"),
    ("Need an errand runner to photocopy and bind my project at the library", "job_okay"),
    ("Content writer needed for a blog post 800 words about campus life", "job_okay"),
    ("Statistics tutor needed for regression analysis help", "job_okay"),
    ("Need someone to help me practice my French conversation skills", "job_okay"),
    ("Animation needed for a short 30 second explainer video", "job_okay"),
    ("Looking for a photographer for student ID photos affordable rate", "job_okay"),
    ("Need help setting up my laptop and installing software", "job_okay"),
    ("Someone to help me carry and move my belongings from hostel A to B", "job_okay"),
    ("Caricature artist needed for a birthday card design", "job_okay"),
    ("Need a research assistant to help gather materials for my literature review", "job_okay"),
    ("Looking for someone to teach me Excel for my project data analysis", "job_okay"),
    ("Food delivery from school cafeteria to engineering block needed by 1pm", "job_okay"),
    ("Someone to pick up my laundry from dry cleaner at school gate", "job_okay"),
    ("Need a music student who can compose a short jingle for my event", "job_okay"),
    ("Need a student to help me practice for a debate competition", "job_okay"),
    ("Someone needed to help queue and pay at the bursary on my behalf", "job_okay"),
    ("Mural artist needed to paint a wall design for our departmental week", "job_okay"),
    ("Need someone to help format my references in APA style", "job_okay"),
    ("Looking for someone to help me rehearse a presentation before my defense", "job_okay"),
    ("3D modelling needed for my engineering final year project", "job_okay"),
    ("Need someone to create a simple website using WordPress budget 8k", "job_okay"),
    ("Looking for a student with a car to help me move things off campus", "job_okay"),
    ("Need a tailor to hem my trousers before tomorrow morning", "job_okay"),
    ("Graphic designer needed for a departmental association logo", "job_okay"),
    ("Need a tutor for MTH102 integral calculus before test next week", "job_okay"),
    ("Someone needed to print and bind 5 copies of my project report", "job_okay"),
    ("Looking for someone to help set up a projector for an event tonight", "job_okay"),
    ("Need a student DJ to perform at our hall week event Friday night", "job_okay"),
    ("Looking for a skilled baker to make a birthday cake for Saturday", "job_okay"),
    ("Need someone to help me fill and submit my NYSC registration forms", "job_okay"),
    ("App developer needed to fix a bug in my React Native project", "job_okay"),
    ("Need someone to edit and clean up my academic research paper", "job_okay"),
    ("I need a graphic designer for a flyer", "job_okay"),
    ("Need a designer for campus event flyer budget 5k", "job_okay"),
    ("Looking for a graphic designer budget 3k", "job_okay"),
    ("Need a flyer designed for my event", "job_okay"),
    ("Graphic designer needed for a flyer affordable budget", "job_okay"),
    ("I need a chef to cook for a small event on campus", "job_okay"),
    ("Looking for someone who can cook for our departmental hangout", "job_okay"),
    ("Need a cook for a small gathering this weekend budget 8k", "job_okay"),
    ("I need a graphic designer for a promotional flyer campus event", "job_okay"),
("Need a designer to create a flyer for my birthday event budget 3k", "job_okay"),
("Looking for a graphic designer to make a poster for our department", "job_okay"),
("Need a flyer designed for a campus hangout this weekend budget 5k", "job_okay"),
("I need a chef to cook small chops for a campus event Saturday", "job_okay"),
("Looking for someone who can cook jollof rice for 20 people event", "job_okay"),

    # ============================================================
    # SCAM — Academic malpractice, fraud, theft, harm
    # ============================================================
    ("Help me hack into my classmate portal to change grades", "scam"),
    ("Need hacker to access university portal", "scam"),
    ("I need help to steal my roommate laptop", "scam"),
    ("Assist me to steal a phone from hostel", "scam"),
    ("Earn 50k daily without doing anything just send 2k registration fee", "scam"),
    ("Send money to double your cash investment platform", "scam"),
    ("Looking for someone to help impersonate me during semester exams", "scam"),
    ("Write my semester exam paper for me hire someone exam impersonation", "scam"),
    ("Pay me to get exam questions before time leakage", "scam"),
    ("Help me change my CGPA on the school portal I will pay well", "scam"),
    ("I need exam runz for tomorrow contact me urgent", "scam"),
    ("Looking for c12 expo for tomorrow paper", "scam"),
    ("Need someone to forge my transcript for postgraduate application", "scam"),
    ("Help me bypass school portal login to change my result", "scam"),
    ("Need someone to write my thesis from scratch and submit as mine", "scam"),
    ("Hire someone to take my online exam on my behalf I will pay 20k", "scam"),
    # ---- DATING ----
("Looking for a female companion to spend time with on campus will pay", "scam"),
("Need a girlfriend for a week will pay good money DM me", "scam"),
("Looking for someone to cuddle and watch movies with tonight payment available", "scam"),
("Need a date for my friend's party this weekend will compensate", "scam"),
("Seeking a romantic companion on campus flexible hours good pay", "scam"),
("Looking for a chill partner to hang out with regularly on campus", "scam"),
    ("I need a fake degree certificate urgently", "scam"),
    ("Looking for someone to help me with BVN cloning fraud work", "scam"),
    ("Need help cracking exam password to get questions early", "scam"),
    ("Who can help me change my grade in the student portal will pay", "scam"),
    ("Help me blackmail my ex she owes me money I have compromising photos", "scam"),
    ("Need someone to send a phishing link to my classmates on my behalf", "scam"),
    ("Help me get access to my lecturer email without them knowing", "scam"),
    ("Looking for someone skilled in card fraud carding work available", "scam"),
    ("Paid work for helping with NIN bypass registration", "scam"),
    ("Who can help me get exam expo for GNS will pay well", "scam"),
    ("Looking for someone to take my CBT exam I cannot attend will pay 15k", "scam"),
    ("I need someone to impersonate me for a job interview I cannot attend", "scam"),
    ("Help needed to access a locked iCloud account that is not mine", "scam"),
    # Add these to your scam section in train.py
("Need someone to sit with me during my exam and help me answer", "scam"),
("Looking for someone to complete my online course modules for me", "scam"),
("Need a typist to write my assignment from scratch not from notes", "scam"),
("Someone to help me answer my take home exam questions tonight", "scam"),
("Need someone to login to my school portal and submit something for me", "scam"),
("Help needed to get into a WhatsApp group of my lecturer", "scam"),
    ("Need someone to help me rob the school bookshop after hours tonight", "scam"),
    ("Looking for a hacker to retrieve money from a scam platform", "scam"),
    ("Need help to get into someone WhatsApp account without them knowing", "scam"),
    ("Someone to help me submit fake receipts to my school finance office", "scam"),
    ("Need someone to create a fake student ID card for me", "scam"),
    ("Help me write a fake recommendation letter from my lecturer", "scam"),
    ("Looking for someone to help me manipulate my school portal grades", "scam"),

    # ============================================================
    # SPAM — Off-platform, nonsense, ads, redirects, out of scope
    # ============================================================
    ("Join my Telegram channel for free data and crypto drops", "spam"),
    ("Free data bundles click link telegram group", "spam"),
    ("Cheap iPhone for sale DM me on WhatsApp 08012345678", "spam"),
    ("Buy cheap laptops and phones call 09000000000", "spam"),
    ("TEST POST PLS IGNORE ASSSSSSSSSSSSSSSSSSSSSS", "spam"),
    ("asdfghjkl qwerty test test test nothing", "spam"),
    ("Investment opportunity double your money in 2 hours whatsapp me", "spam"),
    ("Crypto exchange drop referral link bonus join group", "spam"),
    ("Selling fairly used mattress contact on WhatsApp", "spam"),
    ("I have second hand textbooks for sale DM me on Instagram", "spam"),
    ("Fresh tomatoes and peppers for sale in hostel B room 12", "spam"),
    ("aaaaaaaaaaaa bbbbbbb cccccc nonsense testing nothing", "spam"),
    ("CLICK THIS LINK TO WIN FREE AIRTIME LIMITED TIME ONLY", "spam"),
    ("Join our ponzi scheme and earn passive income weekly guaranteed", "spam"),
    ("Telegram group for free browsing cheat codes join now", "spam"),
    ("Hookup available tonight DM me for rates", "spam"),
    ("Call 08011112222 for cheap data subscription fastest in Nigeria", "spam"),
    ("Hello hello testing 123 ignore this post please nothing here", "spam"),
    ("Selling my room allocation anyone interested contact me", "spam"),
    ("!!!!! URGENT MAKE MONEY FAST !!!!!! CONTACT ME NOW", "spam"),
    ("Room for rent off campus affordable for students dm me whatsapp", "spam"),
    ("Drop your cash app and I will flip your money guaranteed returns", "spam"),
    ("Buy and sell Bitcoin at the best rates join my WhatsApp group now", "spam"),
    ("xyz abc 000 nothing here ignore me random text filler", "spam"),
    ("Pure water sachet for sale in faculty of science block affordable", "spam"),
    ("Looking for hookup in campus tonight contact me privately", "spam"),
    ("Selling designer clothes at giveaway prices DM on Instagram", "spam"),
    ("Free recharge card giveaway follow my page and repost to win", "spam"),
    ("Drop your number for a good time tonight", "spam"),
    ("Best investment platform in Nigeria 300 percent returns weekly", "spam"),
    ("Join this WhatsApp group for verified job opportunities abroad", "spam"),
    ("Recharge card printing software available cheaply DM me now", "spam"),
    ("I have real estate investment opportunity message me on telegram", "spam"),
    ("Forex trading signals group join for free profitable guaranteed", "spam"),
    ("ahhhh nothing just testing this app ignore this please thanks", "spam"),
]

X, y = zip(*training_data)

# Train/test split for honest evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = make_pipeline(
    TfidfVectorizer(
        ngram_range=(1, 3),
        sublinear_tf=True,
        strip_accents='unicode',
        lowercase=True,
        min_df=1,
        max_features=8000
    ),
    LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',   # handles class imbalance
        solver='lbfgs',
      
        random_state=42
    )
)

pipeline.fit(X_train, y_train)

# Evaluate on held-out test set
y_pred = pipeline.predict(X_test)
print("\n=== Test Set Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["job_okay", "scam", "spam"]))

# 5-fold cross-validation on full dataset
scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_weighted')
print(f"=== 5-Fold CV Weighted F1 ===")
print(f"Mean: {scores.mean():.4f}  |  Std: {scores.std():.4f}")

# Retrain on full dataset before saving
pipeline.fit(X, y)
joblib.dump(pipeline, "step_job_moderator.joblib")
print("\nModel saved as 'step_job_moderator.joblib'")