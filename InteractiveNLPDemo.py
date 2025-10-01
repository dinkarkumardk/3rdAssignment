"""
Interactive Natural Language Delivery Analytics System
====================================================

This system demonstrates how to handle ANY natural language question about delivery data
using NLP techniques for intent recognition, entity extraction, and dynamic query processing.

Author: Delivery Analytics Team
Date: September 30, 2025
"""

from NLPDeliveryAnalytics import NLPDeliveryAnalytics


def interactive_demo():
    """
    Interactive demonstration of the NLP system that can handle any human language query
    """
    print("🤖 INTERACTIVE NATURAL LANGUAGE DELIVERY ANALYTICS")
    print("=" * 60)
    print("This system can understand and answer ANY question about delivery data!")
    print("Examples of questions you can ask:")
    print("• 'Why are my orders failing in Delhi?'")
    print("• 'Which city has the highest delivery success rate?'")
    print("• 'How much revenue did we lose due to weather issues?'")
    print("• 'What's the impact of adding 50,000 orders next month?'")
    print("• 'Compare warehouse performance between June and July'")
    print("• 'Show me delivery trends during monsoon season'")
    print()

    # Initialize the system
    analytics = NLPDeliveryAnalytics()

    # Load data
    print("🔄 Initializing system...")
    if not analytics.load_and_aggregate_data():
        print("❌ Failed to load data. Please check your CSV files.")
        return

    print("\n✅ System ready! You can now ask any question about delivery data.")
    print("Type 'quit' or 'exit' to end the session.\n")

    # Demonstration queries to show NLP capabilities
    demo_queries = [
        "What causes most delivery failures in Chennai?",
        "How much money are we losing due to stockouts?",
        "Which drivers have the best performance this month?",
        "Predict the impact of 25000 new orders",
        "Why are weekend deliveries failing more often?",
        "Compare traffic vs weather impact on delays",
        "Show me the worst performing warehouses",
        "What's our customer satisfaction in Mumbai?"
    ]

    print("🎯 DEMONSTRATION OF NLP CAPABILITIES:")
    print("Let me show you how the system handles various types of questions...\n")

    for i, query in enumerate(demo_queries[:4], 1):  # Show first 4 for demo
        print(f"{'='*15} DEMO QUERY {i} {'='*15}")
        print(f"❓ Question: '{query}'")
        print("-" * 50)

        try:
            result = analytics.analyze_query_dynamically(query)
            print(result)
        except Exception as e:
            print(f"❌ Error processing query: {e}")

        print("\n" + "🔸" * 20 + "\n")

    # Interactive mode
    print("🔄 ENTERING INTERACTIVE MODE")
    print("Now you can ask your own questions!")
    print("-" * 40)

    while True:
        try:
            user_query = input("\n❓ Your Question: ").strip()

            if user_query.lower() in ['quit', 'exit', 'bye', 'stop']:
                print("\n👋 Thank you for using the NLP Delivery Analytics System!")
                print("The system successfully demonstrated its ability to understand")
                print("and respond to natural language queries about delivery data.")
                break

            if not user_query:
                print("Please enter a question or type 'quit' to exit.")
                continue

            print("\n" + "="*60)
            result = analytics.analyze_query_dynamically(user_query)
            print(result)
            print("="*60)

            # Suggest follow-up questions
            print("\n💡 SUGGESTED FOLLOW-UP QUESTIONS:")
            print("• 'Tell me more about the top failure reason'")
            print("• 'How can we improve this performance?'")
            print("• 'What's the financial impact of this issue?'")
            print("• 'Compare this with last month's data'")

        except KeyboardInterrupt:
            print("\n\n👋 Session ended by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try rephrasing your question.")


if __name__ == "__main__":
    interactive_demo()
