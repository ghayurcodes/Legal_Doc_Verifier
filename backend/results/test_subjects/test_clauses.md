# RoBERTa Unfair Clause Detector — Test Subjects
# Use these clauses to manually test the model via the frontend.
# Paste each clause into the "Contract Scan" text box and verify the result.

========================================================================
TEST RESULTS SUMMARY (Short Clauses — ~1-2 sentences)
========================================================================
Model scored 8/10 (80%) on these 10 short clauses.

------------------------------------------------------------------------
[1] EXPECTED: SAFE | RESULT: SAFE (0.162) — Standard payment
------------------------------------------------------------------------
You agree to pay a monthly subscription fee of 9.99 dollars. Payments
are due on the 1st of each month. A 14-day free trial is available for
new users.

------------------------------------------------------------------------
[2] EXPECTED: SAFE | RESULT: UNFAIR (0.994) — Governing law [MODEL MISS]
------------------------------------------------------------------------
This Agreement shall be governed by the laws of England and Wales. Any
disputes shall be resolved by the courts of England and Wales.

NOTE: Known model weakness — jurisdiction language is a dataset bias.

------------------------------------------------------------------------
[3] EXPECTED: SAFE | RESULT: SAFE (0.162) — Refund policy
------------------------------------------------------------------------
If you are not satisfied with the product, you may request a full refund
within 30 days of purchase. No questions asked.

------------------------------------------------------------------------
[4] EXPECTED: SAFE | RESULT: SAFE (0.166) — Data privacy
------------------------------------------------------------------------
We collect only the data necessary to provide the service. We do not sell
your personal data to third parties. You may request deletion of your
data at any time.

------------------------------------------------------------------------
[5] EXPECTED: SAFE | RESULT: SAFE (0.189) — Termination with notice
------------------------------------------------------------------------
Either party may terminate this agreement by providing 30 days written
notice to the other party. All outstanding fees will be settled upon
termination.

------------------------------------------------------------------------
[6] EXPECTED: UNFAIR | RESULT: UNFAIR (0.994) — Unilateral termination
------------------------------------------------------------------------
The Company may terminate your account at any time, for any reason or no
reason, without prior notice and without any obligation to refund any
fees paid.

------------------------------------------------------------------------
[7] EXPECTED: UNFAIR | RESULT: SAFE (0.139) — Class action waiver [MODEL MISS]
------------------------------------------------------------------------
You waive your right to bring or participate in any class action lawsuit
or class-wide arbitration against the Company.

NOTE: Short class-action clauses are underrepresented in training data.

------------------------------------------------------------------------
[8] EXPECTED: UNFAIR | RESULT: UNFAIR (0.994) — Unilateral modification
------------------------------------------------------------------------
We reserve the right to change these terms at any time without notice.
Your continued use of the service constitutes your acceptance of the
modified terms.

------------------------------------------------------------------------
[9] EXPECTED: UNFAIR | RESULT: UNFAIR (0.994) — Liability exclusion
------------------------------------------------------------------------
To the fullest extent permitted by law, the Company shall not be liable
for any damages whatsoever, including loss of profits, data, or any
consequential damages.

------------------------------------------------------------------------
[10] EXPECTED: UNFAIR | RESULT: UNFAIR (0.993) — Mandatory arbitration
------------------------------------------------------------------------
All disputes arising out of or related to this agreement shall be resolved
exclusively through binding arbitration. You expressly give up your right
to a jury trial.

========================================================================
LONG CLAUSES (100+ words) — See test results below
========================================================================

------------------------------------------------------------------------
[L1] EXPECTED: SAFE — Comprehensive Service Agreement
------------------------------------------------------------------------
This Service Agreement ("Agreement") is entered into between the Customer
and the Company, and governs the use of all services provided. The Company
agrees to deliver the agreed services in a professional and timely manner,
in accordance with the specifications set out in Schedule A. Either party
may terminate this Agreement upon providing sixty (60) days written notice
to the other party. Upon termination, the Company shall deliver all work
completed to date, and the Customer shall pay for all work satisfactorily
completed up to the termination date. Any disputes arising under this
Agreement shall first be subject to good-faith mediation before either
party may resort to legal proceedings.

------------------------------------------------------------------------
[L2] EXPECTED: SAFE — Data Handling and GDPR Compliance
------------------------------------------------------------------------
We are committed to protecting your personal data in accordance with the
General Data Protection Regulation (GDPR) and applicable local data
protection laws. We collect only the minimum data necessary to provide
our services and retain it only for as long as required. You have the right
to access, correct, or delete your personal data at any time by contacting
our Data Protection Officer. We will never sell, rent, or share your
personal data with third parties for marketing purposes without your
explicit prior consent. All data is stored on encrypted servers located
within the European Economic Area, and we conduct regular security audits
to ensure your data remains safe.

------------------------------------------------------------------------
[L3] EXPECTED: UNFAIR — Predatory All-in-One Terms
------------------------------------------------------------------------
By accessing or using this platform in any manner, including merely
browsing the website, you unconditionally agree to be bound by these Terms
of Service and all future modifications thereof, which we may make at any
time and at our sole discretion without any obligation to notify you.
The Company reserves the right to suspend, restrict, or permanently
terminate your account at any time, for any reason, including reasons
we are not required to disclose, without prior notice and without
liability for any resulting loss of data, content, or access. You
expressly and irrevocably waive any right to seek injunctive or other
equitable relief, and you agree that your sole remedy for any dispute
shall be monetary damages not exceeding the amount you paid in the
prior three months.

------------------------------------------------------------------------
[L4] EXPECTED: UNFAIR — Hidden Liability Exclusion Clause
------------------------------------------------------------------------
To the maximum extent permitted under applicable law, the Company, its
officers, directors, employees, agents, licensors, and service providers
shall have no liability whatsoever arising out of or in connection with
your use of or inability to use the service, including but not limited to
any direct, indirect, incidental, special, punitive, or consequential
damages, loss of profits, loss of revenue, loss of goodwill, loss of data,
business interruption, or the cost of substitute services, even if the
Company has been advised of the possibility of such damages. Some
jurisdictions do not allow the exclusion of implied warranties or limitation
of incidental or consequential damages, so the above limitations may not
apply to you, but the Company shall apply these limitations to the fullest
extent legally permissible.

------------------------------------------------------------------------
[L5] EXPECTED: UNFAIR — Forced Arbitration with Waiver Stack
------------------------------------------------------------------------
Any and all disputes, claims, or controversies arising out of or relating
to this Agreement, the breach, termination, enforcement, or validity
thereof, or your use of the Service, shall be resolved solely by binding
individual arbitration and shall not be consolidated with any other
arbitration or proceeding involving any other party. You expressly waive
your right to a jury trial and your right to participate as a plaintiff or
class member in any purported class action or representative proceeding.
The arbitration shall be conducted by a single arbitrator appointed by the
Company, and the arbitrator's decision shall be final and binding and may
be entered as a judgment in any court of competent jurisdiction. You agree
that any claim must be brought within one (1) year of the event giving
rise to such claim, or it shall be forever barred.
