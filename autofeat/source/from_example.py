import datetime

import faker
import numpy
import polars

from autofeat.convert.into_columns import into_columns
from autofeat.dataset import Dataset
from autofeat.table import Table

_RNG = numpy.random.Generator(numpy.random.PCG64())


_FAKE = faker.Faker()


def from_example(
    *,
    num_accounts: int = 250,
) -> Dataset:
    """Load from randomized example data.

    :param num_accounts: Number of accounts to generate.
    :return: Example dataset.
    """
    accounts = _generate_accounts(num_accounts)
    sessions = _generate_sessions(accounts)
    feedback = _generate_feedback(accounts)

    return Dataset([
        Table(
            name="Accounts",
            data=accounts.lazy(),
            columns=into_columns(accounts),
        ),
        Table(
            name="Sessions",
            data=sessions.lazy(),
            columns=into_columns(sessions),
        ),
        Table(
            name="Feedback",
            data=feedback.lazy(),
            columns=into_columns(feedback),
        ),
    ])


def _generate_accounts(
    num_accounts: int,
) -> polars.DataFrame:
    accounts = []

    for i in range(1, num_accounts + 1):
        company_name = _FAKE.company()
        company_id = f"COMP{i:03d}"
        first_name = _FAKE.first_name()
        last_name = _FAKE.last_name()
        account_owner = f"{first_name} {last_name}"
        email = f"{first_name}.{last_name}@{company_name.replace(' ', '')}.com".lower()
        seats_available = int(_RNG.integers(2, 21))
        contract_start_date_offset = datetime.timedelta(days=int(_RNG.integers(0, 366)))
        contract_start_date = datetime.datetime(2023, 1, 1) + contract_start_date_offset
        contract_size = int(_RNG.integers(1, 11)) * 1000
        churned = _RNG.random() < 0.2
        start_date_offset = datetime.timedelta(days=int(_RNG.integers(0, 31)))
        start_date = contract_start_date - start_date_offset
        renewal_or_churn_date = contract_start_date + datetime.timedelta(days=365)

        accounts.append({
            "Account Owner Email": email,
            "Account Owner": account_owner,
            "Churned": churned,
            "Company ID": company_id,
            "Company Name": company_name,
            "Contract Size per Year": contract_size,
            "Contract Start Date": contract_start_date,
            "Renewal or Churn Date": renewal_or_churn_date,
            "Seats Available": seats_available,
            "Start Date": start_date,
        })

    df = polars.DataFrame(accounts)
    return df.sort("Company ID")


def _generate_sessions(
    accounts: polars.DataFrame,
) -> polars.DataFrame:
    sessions = []

    for account in accounts.rows(named=True):
        # churned accounts tend to actively use fewer of the available seats
        active_usage = _RNG.beta(2, 5) if account["Churned"] else _RNG.beta(5, 2)
        active_seats = _RNG.integers(0, int(active_usage * account["Seats Available"]) + 1)
        active_users = [f"{account['Company ID']}_USER{i:02d}" for i in range(1, active_seats + 1)]

        # all usage happens within a year of renewal_or_churn_date
        usage_start_date = max(
            account["Start Date"],
            account["Renewal or Churn Date"] - datetime.timedelta(days=365),
        )

        usage_period = (account["Renewal or Churn Date"] - usage_start_date).days

        for user_id in active_users:
            # churned users tend to have fewer sessions
            max_sessions = (_RNG.beta(2, 5) if account["Churned"] else _RNG.beta(5, 2)) * 90
            num_sessions = max(0, int(_RNG.normal(max_sessions / 2, max_sessions / 4)))

            for _ in range(num_sessions):
                session_starts = []
                for _ in range(num_sessions):
                    # churned sessions tend to have started earlier in the usage period
                    session_start_offset = (
                        int(_RNG.beta(1, 4) * usage_period)
                        if account["Churned"]
                        else int(_RNG.beta(1.2, 1) * usage_period)
                    )

                    session_start = usage_start_date + datetime.timedelta(days=session_start_offset)
                    session_starts.append(session_start)

                for session_start in sorted(session_starts):
                    session_id = f"{account['Company ID']}_SESSION{_RNG.integers(1, 1001):04d}"
                    session = _generate_session(is_churned=account["Churned"])

                    event_timestamp = session_start
                    for page, event, event_offset in session:
                        event_timestamp += datetime.timedelta(seconds=event_offset)
                        if event_timestamp > account["Renewal or Churn Date"]:
                            break

                        sessions.append({
                            "Company ID": account["Company ID"],
                            "Event": event,
                            "Page Name": page,
                            "Session ID": session_id,
                            "Timestamp": event_timestamp,
                            "User ID": user_id,
                        })

    df = polars.DataFrame(sessions)
    return df.sort("Timestamp")


def _generate_session(
    *,
    is_churned: bool,
) -> list[tuple[str, str, int]]:
    session = []

    # begin session on the home page
    pages = ["home_page", "report_generation", "data_tables"]
    page = pages[0]
    session.append((page, "enter", int(_RNG.integers(1, 10000))))

    # 40% chance to immediately end session if churned, 20% otherwise
    if _RNG.random() < (0.4 if is_churned else 0.2):
        return session

    while len(session) < _RNG.integers(1, 7):
        if _RNG.random() < 0.7:
            # 70% chance of navigating to another page
            session.append((page, "click_button", int(_RNG.integers(10, 20000))))
            page = _RNG.choice([p for p in pages if p != page])
            session.append((page, "enter", int(_RNG.integers(1, 10000))))
        elif _RNG.random() < (0.3 if is_churned else 0.9):
            # 30% chance to start a task if churned, 90% otherwise
            session.append((page, "start_task", int(_RNG.integers(25, 30000))))
        else:
            # end session
            break

    return session



def _generate_feedback(
    accounts: polars.DataFrame,
) -> polars.DataFrame:
    feedback_entries = []

    # Preload ~8 feedback strings from each bucket and sentiment
    for category in FEEDBACK_BUCKETS:
        for sentiment in FEEDBACK_BUCKETS[category]:
            feedback_list = FEEDBACK_BUCKETS[category][sentiment]
            if len(feedback_list) > 8:
                FEEDBACK_BUCKETS[category][sentiment] = _RNG.choice(feedback_list, size=8, replace=False).tolist() #noqa

    for account in accounts.rows(named=True):
        # Decide number of tickets using a beta distribution
        max_tickets = 10  # Maximum number of tickets

        if account["Churned"]:
            # Churned accounts are more likely to have higher number of tickets
            num_tickets = int(_RNG.beta(2, 1) * max_tickets)
        else:
            # Non-churned accounts are more likely to have lower number of tickets
            num_tickets = int(_RNG.beta(1, 2) * max_tickets)

        num_tickets = max(0, min(num_tickets, max_tickets))  # Ensure within bounds

        for _ in range(num_tickets):
            # Generate timestamp between Start Date and Renewal or Churn Date
            start_date = account["Start Date"]
            end_date = account["Renewal or Churn Date"]
            delta_days = (end_date - start_date).days
            random_offset = datetime.timedelta(days=int(_RNG.integers(0, delta_days + 1)))
            timestamp = start_date + random_offset

            # Decide which bucket and sentiment to pick from
            # Churned accounts are 50% more likely to pick negative feedback
            categories = list(FEEDBACK_BUCKETS.keys())
            category = _RNG.choice(categories)

            sentiments = list(FEEDBACK_BUCKETS[category].keys())

            if account["Churned"]:
                # 60% chance to pick negative sentiment
                if _RNG.random() < 0.6 and "Negative" in sentiments:
                    sentiment = "Negative"
                else:
                    sentiment = _RNG.choice(sentiments)
            else:
                # 60% chance to pick neutral or positive sentiment
                if _RNG.random() < 0.6:
                    possible_sentiments = [s for s in sentiments if s != "Negative"]
                    if possible_sentiments:
                        sentiment = _RNG.choice(possible_sentiments)
                    else:
                        sentiment = _RNG.choice(sentiments)
                else:
                    sentiment = _RNG.choice(sentiments)

            # Select feedback from the preloaded feedback strings
            feedback_options = FEEDBACK_BUCKETS[category][sentiment]
            #adding random text at the end so it doesn't get coded as categorical
            email_signature = f"\n\nBest regards,\n{_FAKE.name()}"
            feedback_text = f"{_RNG.choice(feedback_options)}, {_FAKE.sentence(nb_words=3)} {email_signature}" #noqa

            feedback_entries.append({
                "Company ID": account["Company ID"],
                "Timestamp": timestamp,
                "Feedback": feedback_text,
            })

    df = polars.DataFrame(feedback_entries)
    return df.sort("Timestamp")



# Feedback buckets from the JSON data
FEEDBACK_BUCKETS = {
    "Pricing Feedback": {
        "Neutral": [
            "Can you explain what features are included in the Standard plan?",
            "What is the difference between the Basic and Pro tiers?",
            "Do you offer any discounts for annual subscriptions?",
            "Are there any hidden fees associated with the Premium plan?",
            "Is there a trial period available for new users?",
            "How does your pricing compare to other similar services?",
            "Can I upgrade my plan at any time?",
            "Does the Basic plan include customer support?",
            "Are there any limits on data usage with the Standard tier?",
            "What payment methods do you accept?",
            "Is there a setup fee for new accounts?",
            "Can I customize my plan to include specific features?",
            "Are there any additional costs for add-on services?",
            "Does the Pro plan include access to all features?",
            "Is the pricing per user or per organization?",
            "What happens if I exceed the usage limits of my plan?",
            "Do you offer volume discounts for large teams?",
            "Can I downgrade my subscription if needed?",
            "Are updates included in the subscription price?",
            "Is there a difference in support levels between the plans?"
        ],
        "Negative": [
            "The pricing seems a bit steep for the features offered.",
            "I think your service is overpriced compared to competitors.",
            "I can't justify the cost for the Pro plan.",
            "The subscription fees are too high for small businesses.",
            "I find the pricing structure to be too expensive.",
            "The Premium tier is beyond our budget.",
            "Your plans are not affordable for startups.",
            "Even the Basic plan is too costly.",
            "I feel like I'm not getting enough value for the price.",
            "The cost doesn't align with the benefits we receive.",
            "We might have to look elsewhere due to high pricing.",
            "The expensive plans are a barrier for us.",
            "Your pricing model is prohibitive.",
            "We cannot afford the current subscription rates.",
            "I believe the service is overpriced.",
            "The cost is too high for the limited features.",
            "We expected more features for the price.",
            "The high price point is discouraging.",
            "It's hard to justify the expense.",
            "Your service is out of our price range.",
        ],
    },
    "Usability Feedback": {
        "Neutral": [
            "Where can I find the report generation tool?",
            "How do I set up automated alerts?",
            "Is there a tutorial on using the dashboard?",
            "Can you guide me on how to import data?",
            "How do I customize my user profile?",
            "Is there a way to export reports in PDF format?",
            "How do I change the default settings?",
            "Where can I access the analytics features?",
            "Can I integrate the platform with third-party apps?",
            "How do I add new team members to my account?",
            "Is there an option to schedule reports?",
            "How do I navigate to the settings page?",
            "Can you explain how to use the filter options?",
            "Where is the help section located?",
            "How do I create custom templates?",
            "Is there a feature to track user activity?",
            "How can I reset my password?",
            "How do I update my billing information?",
            "Where can I see my usage statistics?",
            "Is there a search function within the platform?",
        ],
        "Negative": [
            "Why is it so hard to find the report generation tool?",
            "I can't figure out how to set up automated alerts; it's too confusing.",
            "Is there no tutorial on using the dashboard? I'm completely lost.",
            "Importing data shouldn't be this difficult.",
            "How am I supposed to customize my profile when options are missing?",
            "Exporting reports in PDF format isn't working.",
            "Why can't I change the default settings? They keep reverting.",
            "Accessing analytics features is a nightmare.",
            "Integration with third-party apps is overly complicated.",
            "Adding new team members shouldn't be this frustrating.",
            "Scheduling reports is impossible with the current setup.",
            "Navigating to the settings page is needlessly complex.",
            "The filter options are unintuitive; how do I even use them?",
            "The help section is useless; it doesn't answer my questions.",
            "Creating custom templates is a hassle.",
            "Tracking user activity isn't accurate; what's the point?",
            "Resetting my password is problematic; I keep getting errors.",
            "Updating billing information shouldn't be so troublesome.",
            "Viewing usage statistics is more complicated than it needs to be.",
            "Why is there no effective search function within the platform?",
        ],
    },
    "Support Frustration": {
        "Negative": [
            "Your support team isn't addressing my issues.",
            "I'm not getting helpful responses from support.",
            "My questions are being ignored.",
            "Support is not resolving my problems.",
            "I feel like I'm talking to a wall with your support.",
            "No one is giving me clear answers.",
            "Your replies are unhelpful and generic.",
            "Support is taking too long to respond.",
            "My concerns are not being taken seriously.",
            "I'm frustrated with the lack of assistance.",
            "You keep sending me irrelevant information.",
            "I have to repeat myself constantly.",
            "Support isn't following up on my tickets.",
            "I feel neglected by your support team.",
            "My issues remain unresolved despite multiple contacts.",
            "Communication from support is poor.",
            "I need real solutions, not canned responses.",
            "You're not providing the help I need.",
            "Support is not addressing the root of the problem.",
            "I'm disappointed with the lack of effective support.",
        ],
        "Positive": [
            "Thank you for your prompt assistance.",
            "I appreciate the help from your support team.",
            "Your support resolved my issue quickly.",
            "Thanks for the detailed explanation.",
            "Great customer service experience!",
            "Your team was very helpful.",
            "I got the answers I needed, thank you.",
            "Support was friendly and efficient.",
            "Thanks for going above and beyond.",
            "I appreciate the quick resolution.",
            "Your assistance was invaluable.",
            "Thank you for the excellent support.",
            "Problem solved, thanks to your team.",
            "Support was very responsive.",
            "I'm grateful for your help.",
            "Thank you for your patience and guidance.",
            "Excellent assistance from support.",
            "Your help made a big difference.",
            "Support provided the solution I needed.",
            "I'm very satisfied with the support received.",
        ],
    },
}
