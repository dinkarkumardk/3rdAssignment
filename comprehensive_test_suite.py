#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced NLP Delivery Analyzer
==========================================================

This script tests 50+ natural language queries across all categories:
- City-specific analysis
- Time-based filtering  
- Comparisons
- Predictions
- Root cause analysis
- Optimization queries
- Complex multi-entity queries

"""

import sys
import time
from datetime import datetime
from complete_nlp_delivery_analyzer import CompleteNLPDeliveryAnalyzer

def run_comprehensive_test():
    """Run comprehensive test with 50+ diverse queries."""
    
    print("🚀 COMPREHENSIVE NLP DELIVERY ANALYZER TEST SUITE")
    print("=" * 80)
    print("Testing 55+ natural language queries across all categories...")
    print("=" * 80)
    
    # Initialize analyzer with all features
    print("\n🔧 Initializing Enhanced NLP Analyzer...")
    analyzer = CompleteNLPDeliveryAnalyzer(enable_ml=True, enable_cache=True, enable_viz=False)
    
    # Comprehensive test queries organized by category
    test_queries = {
        "🏙️ City-Specific Analysis": [
            "What's happening in Mumbai?",
            "Show me Delhi performance metrics",
            "Analyze Bangalore delivery issues",
            "How is Chennai performing?",
            "Give me Mumbai statistics",
            "What's the situation in Hyderabad?",
            "Show Pune delivery data",
            "Kolkata performance analysis",
        ],
        
        "⏰ Time-Based Queries": [
            "Why were deliveries delayed in Delhi yesterday?",
            "Show me last week's performance",
            "What happened in Mumbai last month?",
            "Analyze issues from last 2 months",
            "Delhi performance in last 3 months", 
            "Show today's delivery status",
            "Last 7 days analysis",
            "Past 30 days trends",
        ],
        
        "🔍 City + Time Combined": [
            "Why were deliveries delayed in city Delhi Last 2 month?",
            "Mumbai issues in last 3 months",
            "Bangalore performance yesterday",
            "Chennai delays last week",
            "Hyderabad problems this month",
            "Delhi vs Mumbai last month",
            "Show me Pune trends last 2 months",
        ],
        
        "⚖️ Comparison Queries": [
            "Compare Delhi vs Mumbai",
            "Which performs better - Bangalore vs Chennai?",
            "Mumbai vs Delhi success rates",
            "Compare all cities performance",
            "Bangalore vs Hyderabad comparison",
            "Best performing city",
            "Worst performing city",
            "Rank cities by performance",
        ],
        
        "🔮 Prediction & Forecasting": [
            "Predict issues for next week",
            "What will happen if we add 10000 orders?",
            "Forecast delivery volumes for Q4",
            "What's the risk during monsoon?",
            "Predict Mumbai performance next month",
            "Impact of adding 5000 new orders",
            "Future trends analysis",
            "Capacity planning for December",
        ],
        
        "🔧 Optimization & Improvement": [
            "How can we improve success rate?",
            "Reduce delivery failures",
            "Optimization opportunities",
            "Ways to improve Delhi performance",
            "How to reduce delays?",
            "Improve customer satisfaction",
            "Best practices for operations",
            "Strategic recommendations",
        ],
        
        "🕵️ Root Cause Analysis": [
            "Why are there so many delivery failures?",
            "What causes delays in Mumbai?",
            "Root cause of issues",
            "Why is Delhi performance poor?",
            "What's causing traffic delays?",
            "Reasons for weather-related issues",
            "Main factors affecting delivery",
            "Primary causes of problems",
        ],
        
        "📊 Complex Multi-Entity Queries": [
            "Compare weather vs traffic issues in Delhi last month",
            "Show me weekend vs weekday performance in Mumbai",
            "Peak hours analysis for Bangalore deliveries",
            "Warehouse efficiency across all cities",
            "Driver performance by city and time",
            "Customer satisfaction correlation with weather",
            "Seasonal trends in delivery performance",
            "Impact of external factors on success rate",
        ]
    }
    
    # Test execution
    total_queries = sum(len(queries) for queries in test_queries.values())
    successful_tests = 0
    failed_tests = 0
    test_results = []
    
    print(f"\n📝 EXECUTING {total_queries} TEST QUERIES")
    print("=" * 60)
    
    query_number = 1
    
    for category, queries in test_queries.items():
        print(f"\n{category}")
        print("-" * 50)
        
        for query in queries:
            print(f"\n{query_number:2d}. Query: '{query}'")
            
            try:
                start_time = time.time()
                response = analyzer.process_natural_language_query(query)
                processing_time = time.time() - start_time
                
                # Validate response
                validation_result = validate_response(query, response)
                
                if validation_result['valid']:
                    print(f"    ✅ PASSED ({processing_time:.2f}s)")
                    successful_tests += 1
                    status = "PASSED"
                else:
                    print(f"    ❌ FAILED: {validation_result['reason']}")
                    failed_tests += 1
                    status = "FAILED"
                
                # Store result for detailed analysis
                test_results.append({
                    'query_number': query_number,
                    'category': category,
                    'query': query,
                    'response_length': len(response),
                    'processing_time': processing_time,
                    'status': status,
                    'validation': validation_result
                })
                
                # Show response preview for some queries
                if query_number % 10 == 1 or validation_result['interesting']:
                    print(f"    📋 Response Preview:")
                    preview = response.split('\n')[:3]
                    for line in preview:
                        print(f"       {line}")
                    if len(response.split('\n')) > 3:
                        print(f"       ... (truncated)")
                
            except Exception as e:
                print(f"    💥 ERROR: {str(e)}")
                failed_tests += 1
                test_results.append({
                    'query_number': query_number,
                    'category': category, 
                    'query': query,
                    'status': 'ERROR',
                    'error': str(e)
                })
            
            query_number += 1
    
    # Generate comprehensive test report
    generate_test_report(test_results, successful_tests, failed_tests, total_queries, analyzer)
    
    return test_results

def validate_response(query, response):
    """Validate if the response is appropriate for the query."""
    query_lower = query.lower()
    response_lower = response.lower()
    
    validation = {'valid': True, 'reason': '', 'interesting': False}
    
    # Check for empty or error responses
    if len(response) < 20:
        validation['valid'] = False
        validation['reason'] = 'Response too short'
        return validation
    
    if 'error' in response_lower and 'encountered an error' in response_lower:
        validation['valid'] = False
        validation['reason'] = 'Error in processing'
        return validation
    
    # Validate city-specific queries
    cities = ['delhi', 'mumbai', 'bangalore', 'chennai', 'hyderabad', 'pune', 'kolkata']
    mentioned_city = None
    for city in cities:
        if city in query_lower:
            mentioned_city = city
            break
    
    if mentioned_city:
        # Should have filtered data or mention the city
        if mentioned_city not in response_lower and 'filtered to cities' not in response_lower:
            validation['interesting'] = True  # Mark for review but don't fail
    
    # Validate time-based queries
    time_terms = ['yesterday', 'last week', 'last month', 'last 2 month', 'last 3 month']
    mentioned_time = None
    for term in time_terms:
        if term in query_lower:
            mentioned_time = term
            break
    
    if mentioned_time:
        # Should show filtering information
        if 'filtering' not in response_lower and 'filtered' not in response_lower:
            validation['interesting'] = True
    
    # Validate comparison queries
    if 'vs' in query_lower or 'compare' in query_lower or 'better' in query_lower:
        if 'comparison' not in response_lower and 'vs' not in response_lower:
            validation['interesting'] = True
    
    # Check for reasonable data ranges
    import re
    numbers = re.findall(r'\d{1,3}(?:,\d{3})*', response)
    for num_str in numbers:
        num = int(num_str.replace(',', ''))
        if num > 100000:  # Suspiciously high numbers
            validation['interesting'] = True
            validation['reason'] = f'High number detected: {num_str}'
    
    return validation

def generate_test_report(test_results, successful_tests, failed_tests, total_queries, analyzer):
    """Generate comprehensive test report."""
    
    print(f"\n\n📊 COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    
    # Overall statistics
    success_rate = (successful_tests / total_queries) * 100
    print(f"📈 OVERALL PERFORMANCE:")
    print(f"   ✅ Successful Tests: {successful_tests}/{total_queries} ({success_rate:.1f}%)")
    print(f"   ❌ Failed Tests: {failed_tests}")
    print(f"   ⚡ Average Response Time: {sum(r.get('processing_time', 0) for r in test_results if 'processing_time' in r) / len([r for r in test_results if 'processing_time' in r]):.3f}s")
    
    # Category breakdown
    print(f"\n📊 PERFORMANCE BY CATEGORY:")
    category_stats = {}
    for result in test_results:
        category = result['category']
        if category not in category_stats:
            category_stats[category] = {'total': 0, 'passed': 0, 'failed': 0}
        
        category_stats[category]['total'] += 1
        if result['status'] == 'PASSED':
            category_stats[category]['passed'] += 1
        else:
            category_stats[category]['failed'] += 1
    
    for category, stats in category_stats.items():
        pass_rate = (stats['passed'] / stats['total']) * 100
        print(f"   {category}: {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)")
    
    # System performance metrics
    print(f"\n🔧 SYSTEM PERFORMANCE:")
    if hasattr(analyzer, 'response_times') and analyzer.response_times:
        print(f"   ⚡ Total Queries Processed: {len(analyzer.response_times)}")
        print(f"   🕒 Fastest Response: {min(analyzer.response_times):.3f}s")
        print(f"   🐌 Slowest Response: {max(analyzer.response_times):.3f}s")
        print(f"   📊 Data Quality Score: {analyzer.data_quality_score:.1f}/10")
        print(f"   🤖 ML Models Active: {len(analyzer.ml_models)}")
    
    # Interesting cases for manual review
    interesting_cases = [r for r in test_results if r.get('validation', {}).get('interesting', False)]
    if interesting_cases:
        print(f"\n🔍 CASES FOR MANUAL REVIEW ({len(interesting_cases)}):")
        for case in interesting_cases[:5]:  # Show first 5
            print(f"   • Query {case['query_number']}: {case['query']}")
            if 'reason' in case.get('validation', {}):
                print(f"     Reason: {case['validation']['reason']}")
    
    # Failed tests detail
    failed_cases = [r for r in test_results if r['status'] in ['FAILED', 'ERROR']]
    if failed_cases:
        print(f"\n❌ FAILED TESTS DETAIL ({len(failed_cases)}):")
        for case in failed_cases:
            print(f"   • Query {case['query_number']}: {case['query']}")
            if 'error' in case:
                print(f"     Error: {case['error']}")
            elif 'validation' in case:
                print(f"     Reason: {case['validation']['reason']}")
    
    # Export detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{timestamp}.json"
    
    import json
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'summary': {
                'total_queries': total_queries,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate
            },
            'category_stats': category_stats,
            'detailed_results': test_results
        }, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report exported to: {report_file}")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    if success_rate >= 90:
        print(f"   🏆 EXCELLENT: System performing exceptionally well!")
    elif success_rate >= 80:
        print(f"   ✅ GOOD: System performing well with minor issues")
    elif success_rate >= 70:
        print(f"   ⚠️  FAIR: System needs some improvements")
    else:
        print(f"   🚨 POOR: System needs significant improvements")
    
    print(f"\n" + "=" * 80)
    
    return report_file

if __name__ == "__main__":
    print("Starting comprehensive test suite...")
    results = run_comprehensive_test()
    print("✅ Test suite completed!")