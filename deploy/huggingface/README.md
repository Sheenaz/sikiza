---
title: Sikiza
emoji: 👂
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Sheng-aware distress triage API (DSA 8401 capstone demo)
---

# Sikiza: Sheng-aware distress triage (demo deployment)

FastAPI service for the DSA 8401 capstone project. Source code, report and model card:
https://github.com/Sheenaz/sikiza

* Interactive API page: `/docs`
* Try `POST /v1/checkin` with `{"text": "I have no money for fees this semester"}`

**Demo only, not a clinical service.** The coping library and Sheng lexicon are unreviewed drafts.
This demo is hosted outside Kenya, so do not send real student data (the production design requires
Kenya-based hosting under the Data Protection Act 2019). Message text is never written to the audit log.
If you are in distress, call Kenya Red Cross on 1199 or Befrienders Kenya on +254 722 178 177.
