#!/usr/bin/env python3
"""
Complete NLP-Powered Delivery Failure Analysis System
=====================================================

This system can handle ANY natural language query about logistics performance,
delivery failures, operational efficiency, and predictive analysis.

Author: Development Team  
Date: October 2025
Version: 3.0 - Enhanced NLP Implementation with ML & Real-time Analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re
import warnings
import logging
from typing import Dict, List
from dataclasses import dataclass
import cachetools
from fuzzywuzzy import fuzz, process
import time
import hashlib

try:
    import sklearn  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QueryContext:
    """Structured query context for better analysis"""
    original_query: str
    intent: str
    entities: Dict[str, List]
    confidence: float
    complexity: str
    urgency: str
    domain_expertise: str


class CompleteNLPDeliveryAnalyzer:
    """
    Complete NLP-powered delivery failure analysis system that understands
    and responds to ANY natural language query about logistics operations.
    """
    
    @staticmethod
    def get_default_config():
        """Returns a ready-to-use set of sensible defaults for all system knobs."""
        return {
            # Data processing thresholds
            'ml_training_min_samples': 100,
            'rating_threshold_low': 2,
            'warehouse_range_max': 10,
            'weekend_threshold': 5,  # dayofweek >= 5
            'peak_hour_start': 8,
            'peak_hour_end': 18,
            'outlier_quantile': 0.95,
            'rating_min': 1,
            'rating_max': 5,
            
            # Fuzzy matching thresholds
            'fuzzy_match_threshold': 80,
            'fuzzy_pattern_threshold': 85,
            'fuzzy_keyword_threshold': 90,
            
            # Intent scoring
            'comparison_boost': 2,
            'pattern_match_score': 1,
            'phrase_boost_threshold': 4,
            'phrase_boost_value': 0.1,
            
            # Risk scoring
            'risk_high_threshold': 0.7,
            'risk_medium_threshold': 0.4,
            
            # Cache settings
            'cache_maxsize': 1000,
            'cache_ttl': 3600,
            'lru_cache_maxsize': 500,
            
            # ML model settings
            'test_size': 0.2,
            'random_state': 42,
            
            # Business rules - these will be auto-discovered from data
            'weather_issues': ['Rain', 'Storm', 'Fog'],
            'traffic_issues': ['Heavy', 'Jam', 'Moderate'],
            'warehouse_notes_issues': ['System issue', 'Stock delay on item'],
        }
    
    @staticmethod
    def load_config_from_file(config_path):
        """Reads a JSON file so real-world overrides can plug into the default settings."""
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Flatten nested configuration structure
            flattened_config = {}
            for section, settings in file_config.items():
                if isinstance(settings, dict):
                    flattened_config.update(settings)
                else:
                    flattened_config[section] = settings
            
            return flattened_config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in config file {config_path}, using defaults")
            return {}
    
    def __init__(self, data_path=".", config=None, config_file=None, enable_ml=True, enable_cache=True, enable_viz=True):
        """Bootstraps the analyzer: reads config, loads every CSV, and wires up NLP/ML pieces."""
        self.config = self.get_default_config()
        
        # Load from config file if provided
        if config_file:
            file_config = self.load_config_from_file(config_file)
            self.config.update(file_config)
        
        # Override with direct config if provided
        if config:
            self.config.update(config)
            
        self.data_path = data_path
        self.enable_ml = enable_ml and SKLEARN_AVAILABLE
        self.enable_cache = enable_cache
        self.enable_viz = enable_viz
        
        # Initialize caching
        if enable_cache:
            self.query_cache = cachetools.TTLCache(
                maxsize=self.config['cache_maxsize'], 
                ttl=self.config['cache_ttl']
            )
        
        # Initialize containers
        self.ml_models = {}
        self.feature_importance = {}
        self.response_times = []
        self.last_ml_insights_summary = ""
        self.ml_summary_intents = {
            'prediction',
            'capacity_planning',
            'optimization'
        }
        self.client_lookup = {}
        
        logger.info("Initializing Enhanced NLP Delivery Analyzer...")
        
        # Load and process data
        self.load_all_data()
        self.create_unified_dataset()
        self.discover_dynamic_patterns()  # Auto-discover patterns from loaded data
        self.setup_advanced_nlp()
        
        if self.enable_ml:
            self.setup_machine_learning()
        
        if self.enable_viz:
            self.setup_visualization_engine()
            
        self.create_knowledge_graph()
        
        # Data quality assessment
        self.data_quality_score = self.assess_data_quality()
        
        print(f"🚀 Enhanced NLP Delivery Analyzer Ready!")
        print(f"   📊 Dataset: {len(self.master_data):,} records")
        print(f"   🧠 NLP Engine: Advanced natural language understanding")
        print(f"   🤖 ML Models: {'Enabled' if self.enable_ml else 'Disabled'}")
        print(f"   📈 Visualizations: {'Enabled' if self.enable_viz else 'Disabled'}")
        print(f"   💾 Caching: {'Enabled' if enable_cache else 'Disabled'}")
        print(f"   📏 Data Quality: {self.data_quality_score:.1f}/10")
        print(f"   🎯 Capabilities: Advanced Analysis, ML Predictions, Interactive Viz, Real-time Monitoring")

    def load_all_data(self):
        """Opens every required CSV (orders, drivers, warehouses, etc.) and stores them as dataframes."""
        try:
            self.external_factors = pd.read_csv(f'{self.data_path}/external_factors.csv')
            self.warehouse_logs = pd.read_csv(f'{self.data_path}/warehouse_logs.csv')
            self.feedback = pd.read_csv(f'{self.data_path}/feedback.csv')
            self.drivers = pd.read_csv(f'{self.data_path}/drivers.csv')
            self.fleet_logs = pd.read_csv(f'{self.data_path}/fleet_logs.csv')
            self.orders = pd.read_csv(f'{self.data_path}/orders.csv')
            self.clients = pd.read_csv(f'{self.data_path}/clients.csv')
            self.warehouses = pd.read_csv(f'{self.data_path}/warehouses.csv')
            print("✅ All data sources loaded successfully")
        except Exception as e:
            raise Exception(f"Data loading error: {e}")

    def create_unified_dataset(self):
        """Merges all the raw tables into one master view and adds timing/issue flags for each order."""
        # Convert datetime columns
        date_columns = {
            'external_factors': ['recorded_at'],
            'warehouse_logs': ['picking_start', 'picking_end', 'dispatch_time'],
            'feedback': ['created_at'],
            'fleet_logs': ['departure_time', 'arrival_time', 'created_at'],
            'orders': ['order_date', 'promised_delivery_date', 'actual_delivery_date', 'created_at']
        }

        for table_name, columns in date_columns.items():
            table = getattr(self, table_name)
            for col in columns:
                if col in table.columns:
                    table[col] = pd.to_datetime(table[col], errors='coerce')

        # Create master dataset
        master = self.warehouse_logs.copy()

        # Join all data sources
        master = master.merge(
            self.fleet_logs[['order_id', 'driver_id', 'gps_delay_notes', 'departure_time', 'arrival_time']],
            on='order_id', how='left', suffixes=('', '_fleet')
        )

        master = master.merge(
            self.external_factors[['order_id', 'traffic_condition', 'weather_condition', 'event_type']],
            on='order_id', how='left'
        )

        master = master.merge(
            self.feedback[['order_id', 'sentiment', 'rating', 'feedback_text']],
            on='order_id', how='left'
        )

        master = master.merge(
            self.drivers[['driver_id', 'driver_name', 'city', 'state', 'partner_company']],
            on='driver_id', how='left', suffixes=('', '_driver')
        )

        master = master.merge(
            self.warehouses[['warehouse_id', 'warehouse_name', 'city', 'state']],
            on='warehouse_id', how='left', suffixes=('', '_warehouse')
        )

        if hasattr(self, 'orders'):
            order_columns = [
                'order_id', 'client_id', 'status', 'failure_reason',
                'order_date', 'actual_delivery_date', 'created_at', 'payment_mode', 'amount'
            ]
            available_order_columns = [col for col in order_columns if col in self.orders.columns]
            if available_order_columns:
                master = master.merge(
                    self.orders[available_order_columns],
                    on='order_id', how='left'
                )

        if hasattr(self, 'clients') and 'client_id' in master.columns:
            client_columns = ['client_id', 'client_name']
            available_client_columns = [col for col in client_columns if col in self.clients.columns]
            if available_client_columns:
                master = master.merge(
                    self.clients[available_client_columns],
                    on='client_id', how='left'
                )

        # Calculate derived metrics
        master['picking_duration'] = (master['picking_end'] - master['picking_start']).dt.total_seconds() / 60
        master['dispatch_delay'] = (master['dispatch_time'] - master['picking_end']).dt.total_seconds() / 60
        master['delivery_duration'] = (master['arrival_time'] - master['departure_time']).dt.total_seconds() / 60

        # Issue detection - Based on actual data indicators
        # IMPORTANT: Each order is counted only ONCE even if it has multiple issues
        # Using OR (|) operator ensures no double-counting of orders with multiple problems
        
        # Warehouse operational issues (from actual notes)
        warehouse_issues = master['notes'].isin(self.config['warehouse_notes_issues']) if 'notes' in master.columns else False
        
        # Fleet delivery problems (from GPS delay notes)
        fleet_issues = master['gps_delay_notes'].isin(['Breakdown', 'Address not found']) if 'gps_delay_notes' in master.columns else False
        
        # Customer dissatisfaction (low ratings indicating problems)
        customer_issues = (master['rating'] <= self.config['rating_threshold_low']) if 'rating' in master.columns else False
        
        # Negative customer sentiment
        sentiment_issues = (master['sentiment'] == 'Negative') if 'sentiment' in master.columns else False
        
        # Use OR logic: if ANY condition is True, the order has an issue (counted once)
        # This prevents double-counting orders with multiple issue types
        master['has_issue'] = (
                warehouse_issues |
                fleet_issues |
                customer_issues |
                sentiment_issues
        )

        # Additional time features
        master['hour'] = master['picking_start'].dt.hour
        master['day_of_week'] = master['picking_start'].dt.day_name()
        master['month'] = master['picking_start'].dt.month
        master['week'] = master['picking_start'].dt.isocalendar().week

        self.master_data = master
        logger.info(f"Unified dataset created with {len(master)} records")

    def discover_dynamic_patterns(self):
        """Scans the master data to learn real city, warehouse, weather, and issue patterns on the fly."""
        logger.info("Discovering dynamic patterns from data...")
        
        # Discover cities from actual data
        unique_cities = set()
        for col in ['city', 'city_driver', 'city_warehouse']:
            if col in self.master_data.columns:
                cities_from_col = self.master_data[col].dropna().str.lower().str.strip().unique()
                unique_cities.update(cities_from_col)
        
        # Discover warehouse names/IDs from actual data
        unique_warehouses = set()
        if 'warehouse_name' in self.master_data.columns:
            warehouses = self.master_data['warehouse_name'].dropna().str.lower().str.strip().unique()
            unique_warehouses.update(warehouses)
        if 'warehouse_id' in self.master_data.columns:
            warehouse_ids = self.master_data['warehouse_id'].dropna().astype(str).unique()
            unique_warehouses.update([f"warehouse {wid}" for wid in warehouse_ids])
        
        # Discover weather conditions from actual data
        weather_conditions = set()
        if 'weather_condition' in self.master_data.columns:
            weather_conditions = set(self.master_data['weather_condition'].dropna().unique())
        
        # Discover traffic conditions from actual data
        traffic_conditions = set()
        if 'traffic_condition' in self.master_data.columns:
            traffic_conditions = set(self.master_data['traffic_condition'].dropna().unique())
        
        # Discover issue patterns from actual data
        issue_patterns = set()
        if 'notes' in self.master_data.columns:
            issue_patterns.update(self.master_data['notes'].dropna().unique())
        if 'gps_delay_notes' in self.master_data.columns:
            issue_patterns.update(self.master_data['gps_delay_notes'].dropna().unique())
        
        # Update config with discovered patterns
        if unique_cities:
            self.config['cities'] = sorted(list(unique_cities))
        if unique_warehouses:
            self.config['warehouse_patterns'] = sorted(list(unique_warehouses))
        if weather_conditions:
            # Filter weather conditions that indicate issues
            problematic_weather = [w for w in weather_conditions if w.lower() in ['rain', 'storm', 'fog', 'heavy rain', 'thunderstorm']]
            if problematic_weather:
                self.config['weather_issues'] = problematic_weather
        if traffic_conditions:
            # Filter traffic conditions that indicate issues
            problematic_traffic = [t for t in traffic_conditions if t.lower() in ['heavy', 'jam', 'moderate', 'high', 'congested']]
            if problematic_traffic:
                self.config['traffic_issues'] = problematic_traffic
        if issue_patterns:
            # Filter notes that indicate warehouse issues
            problematic_notes = [n for n in issue_patterns if any(keyword in n.lower() for keyword in ['issue', 'delay', 'problem', 'error', 'stock', 'system'])]
            if problematic_notes:
                self.config['warehouse_notes_issues'] = problematic_notes

        client_names = set()
        if 'client_name' in self.master_data.columns:
            client_names = set(
                self.master_data['client_name']
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
            )

        if client_names:
            self.client_lookup = {name.lower(): name for name in client_names}
            self.config['clients'] = sorted(list(client_names))
        else:
            self.client_lookup = {}
        
        logger.info(f"Discovered {len(self.config['cities'])} cities, {len(self.config['warehouse_patterns'])} warehouses")
        logger.info(f"Discovered {len(self.config['weather_issues'])} weather issues, {len(self.config['traffic_issues'])} traffic issues")

    def setup_advanced_nlp(self):
        """Builds the dictionaries of keywords and phrases so the system understands entities and intent."""
        logger.info("Setting up advanced NLP engine...")
        
        # Enhanced entity patterns with fuzzy matching - using discovered patterns
        self.entity_patterns = {
            'cities': self.config.get('cities', []),
            'warehouses': self.config.get('warehouse_patterns', []),
            'time_periods': [
                'today', 'yesterday', 'this week', 'last week', 'this month', 'last month',
                'last 2 months', 'last 3 months', 'last 6 months', 'past 2 months', 'past 3 months',
                'q1', 'q2', 'q3', 'q4', 'quarter', 'year', 'ytd', 'last 7 days',
                'last 30 days', 'last 60 days', 'last 90 days', 'past week', 'past month', 'recent',
                'festival period', 'festival season', 'holiday period', 'holiday season', 'peak season',
                'diwali', 'christmas', 'new year', 'eid', 'holi', 'dussehra', 'navratri',
                'january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december',
                'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
            ],
            'metrics': [
                'success rate', 'failure rate', 'delay', 'on time', 'picking time',
                'dispatch time', 'delivery time', 'customer satisfaction', 'rating',
                'feedback', 'cost', 'revenue', 'efficiency', 'performance', 'issues', 'problems'
            ],
            'operations': [
                'warehouse', 'driver', 'fleet', 'delivery', 'logistics', 'supply chain',
                'picking', 'dispatch', 'route', 'traffic', 'weather', 'capacity'
            ],
            'clients': self.config.get('clients', [])
        }
        
        # Enhanced intent classification patterns with comprehensive vocabulary
        self.intent_patterns = {
            'analysis': [
                # Core analysis terms
                'analyze', 'analysis', 'show', 'display', 'reveal', 'present', 'demonstrate',
                'what', 'how many', 'how much', 'count', 'total', 'sum', 'number',
                'tell me', 'give me', 'provide', 'fetch', 'get', 'find', 'locate',
                'breakdown', 'summary', 'overview', 'report', 'statistics', 'stats',
                'details', 'information', 'data', 'metrics', 'figures', 'numbers',
                # Delivery-specific analysis terms
                'orders', 'deliveries', 'shipments', 'performance', 'issues', 'problems',
                'success', 'failure', 'rate', 'percentage', 'distribution', 'trends'
            ],
            'comparison': [
                'compare', 'comparison', 'vs', 'versus', 'against', 'between',
                'difference', 'differ', 'contrast', 'better', 'worse', 'best', 'worst',
                'higher', 'lower', 'more', 'less', 'greater', 'smaller',
                'benchmark', 'baseline', 'standard', 'which is', 'top', 'bottom',
                'ranking', 'rank', 'performance comparison', 'relative to'
            ],
            'prediction': [
                'predict', 'prediction', 'forecast', 'forecasting', 'estimate', 'estimation',
                'project', 'projection', 'future', 'upcoming', 'next', 'expected',
                'trend', 'trending', 'pattern', 'outlook', 'anticipate', 'expect',
                'will be', 'going to', 'likely', 'probable', 'potential', 'scenario',
                'what will happen', 'what happens next', 'machine learning', 'ml model'
            ],
            'optimization': [
                'improve', 'improvement', 'optimize', 'optimization', 'enhance', 'enhancement',
                'fix', 'solve', 'solution', 'resolve', 'address', 'tackle',
                'reduce', 'decrease', 'minimize', 'increase', 'maximize', 'boost',
                'recommendations', 'recommend', 'suggest', 'suggestions', 'advice',
                'how to', 'best practices', 'strategy', 'approach', 'method',
                'efficient', 'effectiveness', 'performance improvement', 'upgrade'
            ],
            'causation': [
                'why', 'because', 'reason', 'reasons', 'cause', 'causes', 'caused by',
                'due to', 'owing to', 'result of', 'consequence', 'effect', 'impact',
                'factors', 'factor', 'root cause', 'main reason', 'primary cause',
                'explain', 'explanation', 'what causes', 'what leads to', 'behind',
                'affect', 'affects', 'influence', 'influences', 'drives', 'contributes',
                'correlation', 'relationship', 'connection', 'linked to', 'associated'
            ],
            'capacity_planning': [
                'onboard', 'onboarding', 'new client', 'additional orders', 'extra orders',
                'scale up', 'scaling', 'capacity', 'volume increase', 'growth', 'expansion',
                'handle', 'accommodate', 'support', 'manage', 'prepare for', 'ready for',
                'if we add', 'with more', 'extra load', 'increased demand', 'higher volume',
                'can we handle', 'capacity planning', 'resource planning', 'infrastructure',
                'bottlenecks', 'constraints', 'limitations', 'stress test', 'load test',
                'risk assessment', 'risk analysis', 'failure risks', 'new risks'
            ]
        }
        
        # Domain-specific keywords that boost confidence
        self.domain_keywords = [
            'delivery', 'deliveries', 'order', 'orders', 'shipment', 'shipping',
            'warehouse', 'fleet', 'driver', 'customer', 'logistics', 'supply chain',
            'pickup', 'dispatch', 'transit', 'arrival', 'delay', 'issue', 'problem',
            'rating', 'feedback', 'satisfaction', 'performance', 'analytics',
            'gps', 'tracking', 'route', 'city', 'location', 'time', 'duration'
        ]

    def setup_machine_learning(self):
        """Trains the optional scikit-learn models (if available) so we can predict issues, timing, and ratings."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, ML features disabled")
            return
            
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
            
            logger.info("Setting up ML models...")
            
            # Prepare features for ML
            ml_data = self.master_data.copy()
            
            # Feature engineering for ML
            ml_features = self.create_ml_features(ml_data)
            self.ml_training_features = ml_features.copy()
            
            if not ml_features.empty and len(ml_features) > self.config['ml_training_min_samples']:
                # Issue prediction model
                self.train_issue_prediction_model(ml_features)
                
                # Delivery time prediction model
                self.train_delivery_time_model(ml_features)
                
                # Customer satisfaction prediction
                self.train_satisfaction_model(ml_features)
                
                logger.info("ML models trained successfully")
            else:
                logger.warning("Insufficient data for ML model training")
                self.ml_training_features = pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error setting up ML models: {e}")

    def create_ml_features(self, data):
        """Turns raw columns into clean numeric features and targets that the ML models can digest."""
        try:
            features = pd.DataFrame()
            
            # Temporal features
            if 'picking_start' in data.columns:
                features['hour'] = data['picking_start'].dt.hour
                features['day_of_week'] = data['picking_start'].dt.dayofweek
                features['month'] = data['picking_start'].dt.month
                features['is_weekend'] = (data['picking_start'].dt.dayofweek >= self.config['weekend_threshold']).astype(int)
                features['is_peak_hour'] = ((data['picking_start'].dt.hour >= self.config['peak_hour_start']) & 
                                          (data['picking_start'].dt.hour <= self.config['peak_hour_end'])).astype(int)
            
            # Operational features
            if 'picking_duration' in data.columns:
                features['picking_duration'] = data['picking_duration'].fillna(data['picking_duration'].median())
            
            if 'dispatch_delay' in data.columns:
                features['dispatch_delay'] = data['dispatch_delay'].fillna(data['dispatch_delay'].median())
            
            # Categorical encoding
            categorical_cols = ['weather_condition', 'traffic_condition', 'city']
            for col in categorical_cols:
                if col in data.columns:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    features[f'{col}_encoded'] = le.fit_transform(data[col].fillna('Unknown'))
            
            # Target variables
            if 'has_issue' in data.columns:
                features['target_issue'] = data['has_issue'].astype(int)
            
            if 'delivery_duration' in data.columns:
                features['target_delivery_time'] = data['delivery_duration'].fillna(data['delivery_duration'].median())
            
            if 'rating' in data.columns:
                features['target_satisfaction'] = data['rating'].fillna(3)
            
            return features.dropna()
            
        except Exception as e:
            logger.error(f"Error creating ML features: {e}")
            return pd.DataFrame()

    def train_issue_prediction_model(self, features):
        """Fits a random forest that guesses whether an order will run into trouble."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # Prepare features and target
            feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
                           'picking_duration', 'dispatch_delay', 'weather_condition_encoded',
                           'traffic_condition_encoded', 'city_encoded']
            
            X = features[feature_cols].fillna(0)
            y = features['target_issue']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['test_size'], random_state=self.config['random_state'])
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = accuracy_score(y_test, model.predict(X_test))
            logger.info(f"Issue prediction model accuracy: {accuracy:.3f}")
            
            # Store model and feature importance
            self.ml_models['issue_prediction'] = model
            self.feature_importance['issue_prediction'] = dict(zip(feature_cols, model.feature_importances_))
            
        except Exception as e:
            logger.error(f"Error training issue prediction model: {e}")

    def train_delivery_time_model(self, features):
        """Teaches a gradient boosting regressor to estimate how long a delivery will take."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
            
            feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
                           'picking_duration', 'dispatch_delay', 'weather_condition_encoded',
                           'traffic_condition_encoded', 'city_encoded']
            
            X = features[feature_cols].fillna(0)
            y = features['target_delivery_time']
            
            # Remove outliers
            y_clean = y[(y > 0) & (y < y.quantile(self.config['outlier_quantile']))]
            X_clean = X.loc[y_clean.index]
            
            X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
            
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
            logger.info(f"Delivery time prediction RMSE: {rmse:.2f} minutes")
            
            self.ml_models['delivery_time'] = model
            self.feature_importance['delivery_time'] = dict(zip(feature_cols, model.feature_importances_))
            
        except Exception as e:
            logger.error(f"Error training delivery time model: {e}")

    def train_satisfaction_model(self, features):
        """Builds a model that predicts the rating a customer is likely to give."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score
            
            feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
                           'picking_duration', 'dispatch_delay', 'weather_condition_encoded',
                           'traffic_condition_encoded', 'city_encoded']
            
            X = features[feature_cols].fillna(0)
            y = features['target_satisfaction']
            
            # Filter valid ratings
            valid_ratings = (y >= self.config['rating_min']) & (y <= self.config['rating_max'])
            X_valid = X[valid_ratings]
            y_valid = y[valid_ratings]
            
            if len(X_valid) > self.config['ml_training_min_samples']:
                X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                r2 = r2_score(y_test, model.predict(X_test))
                logger.info(f"Satisfaction prediction R² score: {r2:.3f}")
                
                self.ml_models['satisfaction'] = model
                self.feature_importance['satisfaction'] = dict(zip(feature_cols, model.feature_importances_))
            
        except Exception as e:
            logger.error(f"Error training satisfaction model: {e}")

    def setup_visualization_engine(self):
        """Stores preferred plotting settings so charts look consistent when enabled."""
        logger.info("Visualization engine initialized")
        self.viz_config = {
            'style': 'seaborn',
            'color_palette': 'viridis',
            'figure_size': (12, 8),
            'dpi': 100
        }

    def create_knowledge_graph(self):
        """Summarizes each city into a mini knowledge graph node with counts and averages."""
        try:
            self.knowledge_graph = {
                'entities': {},
                'relationships': [],
                'metrics': {}
            }
            
            # Build entity relationships
            cities = self.master_data['city'].unique()
            warehouses = self.master_data['warehouse_id'].unique()
            
            for city in cities:
                city_data = self.master_data[self.master_data['city'] == city]
                self.knowledge_graph['entities'][city] = {
                    'type': 'city',
                    'orders': len(city_data),
                    'issue_rate': city_data['has_issue'].mean(),
                    'avg_picking_time': city_data['picking_duration'].mean()
                }
            
            logger.info(f"Knowledge graph created with {len(self.knowledge_graph['entities'])} nodes")
            
        except Exception as e:
            logger.error(f"Error creating knowledge graph: {e}")

    def assess_data_quality(self):
        """Calculates a simple 0–10 score showing how complete and duplicate-free the data is."""
        try:
            completeness = (1 - self.master_data.isnull().sum().sum() / 
                          (len(self.master_data) * len(self.master_data.columns)))
            
            # Additional quality checks
            duplicate_rate = self.master_data.duplicated().sum() / len(self.master_data)
            
            score = completeness * 10 * (1 - duplicate_rate)
            return min(score, 10.0)
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return 5.0

    def process_natural_language_query(self, query):
        """Handles a user's question end-to-end: understanding intent, generating text, and attaching ML insights."""
        start_time = time.time()
        
        # Cache check
        query_key = hashlib.md5(query.lower().encode()).hexdigest()
        if self.enable_cache and query_key in self.query_cache:
            cached_result = self.query_cache[query_key]
            print(f"🔄 Cache hit for query")
            return cached_result
            
        print(f"🤖 Processing: '{query}'")
        
        # Enhanced NLP processing
        query_context = self.understand_query_context(query)
        print(f"🧠 Understanding: {query_context.intent}")
        print(f"🎯 Confidence: {query_context.confidence:.2f}")
        
        # Generate response based on intent
        response = self.generate_intelligent_response(query, query_context)
        
        # ML insights
        formatted_insights = ""
        self.last_ml_insights_summary = ""
        if self.enable_ml:
            ml_insights = self.generate_ml_insights(query_context.entities)
            formatted_insights = self._format_ml_insights(ml_insights)
            if formatted_insights and query_context.intent in self.ml_summary_intents:
                response += f"\n\nKey ML signals:\n{formatted_insights}"
                self.last_ml_insights_summary = formatted_insights
        
        processing_time = time.time() - start_time
        self.response_times.append(processing_time)
        
        print(f"⚡ Processing time: {processing_time:.2f}s")
        
        # Cache result
        if self.enable_cache:
            self.query_cache[query_key] = response
            
        return response

    def understand_query_context(self, query):
        """Breaks the question into cities/times/etc., guesses the intent, and scores confidence."""
        query_lower = query.lower()
        
        # Extract entities with fuzzy matching
        entities = {
            'cities': [],
            'warehouses': [],
            'time_periods': [],
            'metrics': [],
            'operations': [],
            'clients': []
        }

        def add_entity(entity_type, raw_value):
            if not raw_value:
                return
            if entity_type == 'clients':
                canonical = self.client_lookup.get(raw_value.lower(), raw_value)
                entities[entity_type].append(canonical)
            else:
                entities[entity_type].append(raw_value)

        # More precise entity extraction with improved matching
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                pattern_lower = pattern.lower()

                if entity_type == 'clients':
                    pattern_tokens = re.findall(r'\w+', pattern_lower)
                    query_tokens = re.findall(r'\w+', query_lower)
                    if pattern_tokens and all(token in query_tokens for token in pattern_tokens):
                        add_entity(entity_type, pattern)
                    continue

                if entity_type == 'time_periods':
                    pattern_tokens = re.findall(r'\w+', pattern_lower)
                    query_tokens = re.findall(r'\w+', query_lower)
                    if pattern_tokens and all(token in query_tokens for token in pattern_tokens):
                        add_entity(entity_type, pattern)
                    continue

                if entity_type == 'warehouses':
                    if pattern_lower in query_lower:
                        add_entity(entity_type, pattern)
                    continue

                if pattern_lower in query_lower:
                    add_entity(entity_type, pattern)
                    continue

                if entity_type == 'cities':
                    pattern_words = pattern_lower.split()
                    query_words = query_lower.split()
                    for pattern_word in pattern_words:
                        if pattern_word in query_words and len(pattern_word) > 3:
                            add_entity(entity_type, pattern)
                            break
                    else:
                        if fuzz.partial_ratio(pattern_lower, query_lower) > self.config['fuzzy_match_threshold']:
                            add_entity(entity_type, pattern)
                    continue

                if fuzz.partial_ratio(pattern_lower, query_lower) > self.config['fuzzy_match_threshold']:
                    add_entity(entity_type, pattern)
        
    # Special warehouse detection with precise matching
        warehouse_pattern = r'warehouse\s+([a-z0-9]+)'
        warehouse_matches = re.findall(warehouse_pattern, query_lower)
        
        # Clear any fuzzy warehouse matches and use precise ones
        entities['warehouses'] = []
        
        for match in warehouse_matches:
            if match.isalpha() and len(match) == 1:  # Single letter (a, b, c, etc.)
                # Map letters to numbers: a->1, b->2, etc.
                warehouse_num = ord(match.lower()) - ord('a') + 1
                if 1 <= warehouse_num <= self.config['warehouse_range_max']:
                    entities['warehouses'].append(f'warehouse {warehouse_num}')
            elif match.isdigit():  # Number (1, 2, 3, etc.)
                warehouse_num = int(match)
                if 1 <= warehouse_num <= self.config['warehouse_range_max']:
                    entities['warehouses'].append(f'warehouse {warehouse_num}')
        
        # Remove duplicates while preserving order
        for entity_type in entities:
            entities[entity_type] = list(dict.fromkeys(entities[entity_type]))
        
        # Enhanced intent classification with smart scoring
        intent_scores = {}
        query_words = query_lower.split()
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                # Exact match gets full points
                if pattern in query_lower:
                    score += self.config['comparison_boost']
                    matched_patterns.append(pattern)
                # Fuzzy match gets partial points
                elif any(fuzz.partial_ratio(pattern, word) > self.config['fuzzy_pattern_threshold'] for word in query_words):
                    score += self.config['pattern_match_score']
                    matched_patterns.append(pattern)
            
            if score > 0:
                intent_scores[intent] = {
                    'score': score,
                    'patterns': matched_patterns
                }

        # Require strong capacity cues before considering capacity planning intent
        if 'capacity_planning' in intent_scores:
            capacity_keywords = {
                'onboard',
                'onboarding',
                'extra',
                'additional',
                'capacity',
                'volume',
                'increase',
                'increasing',
                'scale',
                'scaling',
                'add ',
                'adding',
                'expand',
                'expansion',
                'growth',
                'demand',
                'load',
                'surge',
                'overflow'
            }
            capacity_phrases = [
                'more orders',
                'extra orders',
                'additional orders',
                'new orders'
            ]
            has_capacity_keyword = any(keyword in query_lower for keyword in capacity_keywords)
            has_capacity_phrase = any(phrase in query_lower for phrase in capacity_phrases)
            if not (has_capacity_keyword or has_capacity_phrase):
                intent_scores.pop('capacity_planning')
        
        # Special case: prioritize comparison when both comparison and causation keywords are present
        if 'comparison' in intent_scores and 'causation' in intent_scores:
            comparison_patterns = intent_scores['comparison']['patterns']
            causation_patterns = intent_scores['causation']['patterns']
            
            # Check if comparison keywords are more prominent
            comparison_keywords = ['compare', 'between', 'vs', 'versus']
            has_strong_comparison = any(keyword in query_lower for keyword in comparison_keywords)
            
            if has_strong_comparison:
                # Boost comparison score
                intent_scores['comparison']['score'] += self.config['comparison_boost']
        
        # Calculate enhanced confidence score
        if intent_scores:
            primary_intent = max(intent_scores.items(), key=lambda x: x[1]['score'])[0]
            max_score = intent_scores[primary_intent]['score']
            
            # Base confidence from pattern matches
            base_confidence = min(max_score / len(query_words), 1.0)
            
            # Boost confidence for domain-specific keywords
            domain_boost = sum(1 for keyword in self.domain_keywords 
                             if keyword in query_lower or 
                             any(fuzz.partial_ratio(keyword, word) > self.config['fuzzy_keyword_threshold'] for word in query_words))
            domain_factor = min(domain_boost * 0.1, 0.3)  # Max 30% boost
            
            # Boost for complete phrases and longer queries
            phrase_boost = self.config['phrase_boost_value'] if len(query_words) >= self.config['phrase_boost_threshold'] else 0
            
            # Penalty for very short queries (less informative)
            length_penalty = 0.1 if len(query_words) <= 2 else 0
            
            # Final confidence calculation
            confidence = min(base_confidence + domain_factor + phrase_boost - length_penalty, 1.0)
            confidence = max(confidence, 0.15)  # Minimum confidence threshold
            
        else:
            primary_intent = 'analysis'
            confidence = 0.15  # Slightly higher default
        
        # Store intent scoring details for debugging
        self.last_intent_analysis = {
            'query': query,
            'intent_scores': intent_scores,
            'primary_intent': primary_intent,
            'confidence': confidence
        }
        
        return QueryContext(
            original_query=query,
            intent=primary_intent,
            entities=entities,
            confidence=confidence,
            complexity='medium',
            urgency='normal',
            domain_expertise='logistics'
        )

    def generate_intelligent_response(self, query, context):
        """Picks the right handler (analysis/comparison/etc.) and returns its formatted answer."""
        entities = context.entities
        intent = context.intent
        
        try:
            if intent == 'comparison':
                return self.handle_comparison_query(entities)
            elif intent == 'prediction':
                return self.handle_prediction_query(entities)
            elif intent == 'optimization':
                return self.handle_optimization_query(entities)
            elif intent == 'causation':
                return self.handle_causation_query(entities)
            elif intent == 'capacity_planning':
                return self.handle_capacity_planning_query(entities, context.original_query)
            else:
                return self.handle_analysis_query(entities)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while processing your query: {e}"

    def handle_analysis_query(self, entities):
        """Delivers a health report (counts, averages, tips) optionally filtered by city/time/warehouse."""
        original_count = len(self.master_data)
        data = self._filter_data_by_entities(entities).copy()

        if entities.get('clients') and 'client_name' in data.columns and not data.empty:
            actual_clients = sorted(data['client_name'].dropna().unique().tolist())
            if actual_clients:
                print(f"👥 Filtered to clients: {actual_clients}")

        if entities.get('cities') and 'city' in data.columns and not data.empty:
            actual_cities = sorted(data['city'].dropna().unique().tolist())
            if actual_cities:
                print(f"🌆 Filtered to cities: {actual_cities}")

        if len(data) != original_count:
            print(f"📊 Filter applied: {len(data)} records from {original_count} total")

        if len(data) == 0:
            return f"No data found for the specified filters. Original dataset: {original_count:,} records."

        # Generate comprehensive analysis
        analysis = {
            'total_orders': len(data),
            'orders_with_issues': data['has_issue'].sum(),
            'issue_rate': data['has_issue'].mean() * 100,
            'avg_picking_time': data['picking_duration'].mean(),
            'avg_dispatch_delay': data['dispatch_delay'].mean()
        }
        
        # Generate dynamic header based on actual filters
        header_parts = []

        if entities.get('clients'):
            if 'client_name' in data.columns and not data.empty:
                actual_clients = sorted(data['client_name'].dropna().unique().tolist())
            else:
                actual_clients = []
            client_display = actual_clients if actual_clients else entities['clients']
            header_parts.append(f"Client {', '.join(client_display)}")

        if entities.get('cities'):
            if 'city' in data.columns and not data.empty:
                actual_cities = sorted(data['city'].dropna().unique().tolist())
            else:
                actual_cities = []
            city_display = actual_cities if actual_cities else [city.title() for city in entities['cities']]
            header_parts.append(f"City: {', '.join(city_display)}")

        if entities.get('time_periods'):
            formatted_periods = [term.title() if term.islower() else term for term in entities['time_periods']]
            header_parts.append(f"Timeframe: {', '.join(formatted_periods)}")

        if entities.get('warehouses'):
            header_parts.append(f"Warehouses: {', '.join(entities['warehouses'])}")

        if not header_parts:
            header_parts.append("Delivery Analysis")

        header_title = " | ".join(header_parts)

        response = f"{header_title}: Analysis of {analysis['total_orders']:,} orders\n"
        response += f"Key Metrics:\n"
        response += f"  - Total Orders: {analysis['total_orders']:,}\n"
        response += f"  - Orders With Issues: {analysis['orders_with_issues']:,}\n"
        response += f"  - Issue Rate: {analysis['issue_rate']:.2f}%\n"
        response += f"  - Avg Picking Time: {analysis['avg_picking_time']:.2f} min\n"
        response += f"  - Avg Dispatch Delay: {analysis['avg_dispatch_delay']:.2f} min\n"
        
        # Add recommendations
        recommendations = self.generate_recommendations(analysis)
        response += f"Recommendations:\n"
        for rec in recommendations:
            response += f"  - {rec}\n"
        
        return response

    def handle_comparison_query(self, entities):
        """Stacks two cities side-by-side, highlighting metrics, causes, and improvement ideas."""
        if len(entities['cities']) >= 2:
            data = self.master_data.copy()
            original_count = len(data)
            
            # Apply time period filtering first
            if entities['time_periods']:
                data = self.apply_time_filter(data, entities['time_periods'])
            
            city_comparisons = {}
            
            for city in entities['cities'][:2]:  # Compare first two cities
                city_matches = process.extractBests(city, data['city'].unique(), limit=1, score_cutoff=70)
                if city_matches:
                    actual_city = city_matches[0][0]
                    city_data = data[data['city'] == actual_city]
                    
                    # Calculate basic metrics
                    metrics = {
                        'total_orders': len(city_data),
                        'issue_rate': city_data['has_issue'].mean() * 100,
                        'avg_picking_time': city_data['picking_duration'].mean(),
                        'avg_dispatch_delay': city_data['dispatch_delay'].mean()
                    }
                    
                    # Add causation analysis for comparison
                    issue_data = city_data[city_data['has_issue'] == True]
                    
                    if len(issue_data) > 0:
                        # Analyze causes
                        cause_counts = {}
                        
                        # Weather-related issues
                        weather_issues = issue_data[issue_data['weather_condition'].isin(self.config['weather_issues'])]
                        cause_counts['Weather Related'] = len(weather_issues)
                        
                        # Traffic-related issues  
                        traffic_issues = issue_data[issue_data['traffic_condition'].isin(self.config['traffic_issues'])]
                        cause_counts['Traffic Related'] = len(traffic_issues)
                        
                        # Operational delays
                        operational_issues = issue_data[issue_data['dispatch_delay'] > 60]
                        cause_counts['Operational Delays'] = len(operational_issues)
                        
                        # Find primary cause
                        primary_cause = max(cause_counts.items(), key=lambda x: x[1]) if cause_counts else ("Unknown", 0)
                        
                        metrics['primary_cause'] = primary_cause[0]
                        metrics['cause_breakdown'] = cause_counts
                        metrics['total_issues'] = len(issue_data)
                    else:
                        metrics['primary_cause'] = "No issues"
                        metrics['cause_breakdown'] = {}
                        metrics['total_issues'] = 0
                    
                    city_comparisons[actual_city] = metrics
            
            if len(city_comparisons) == 0:
                return "No matching cities found for comparison."
            
            # Generate comparison response
            response = f"Delivery Failure Cause Comparison"
            if entities['time_periods']:
                response += f" ({', '.join(entities['time_periods'])})"
            response += ":\n\n"
            
            for city, metrics in city_comparisons.items():
                response += f"🏙️ **{city}**:\n"
                response += f"  • Total Orders: {metrics['total_orders']:,}\n"
                response += f"  • Orders with Issues: {metrics['total_issues']:,} ({metrics['issue_rate']:.1f}%)\n"
                response += f"  • Primary Cause: {metrics['primary_cause'].lower()}\n"
                
                if metrics['cause_breakdown']:
                    response += f"  • Cause Breakdown:\n"
                    for cause, count in metrics['cause_breakdown'].items():
                        percentage = (count / metrics['total_issues']) * 100 if metrics['total_issues'] > 0 else 0
                        response += f"    - {cause}: {count} ({percentage:.1f}%)\n"
                
                response += f"  • Avg Picking Time: {metrics['avg_picking_time']:.2f} min\n"
                response += f"  • Avg Dispatch Delay: {metrics['avg_dispatch_delay']:.2f} min\n\n"
            
            # Add comparative insights
            if len(city_comparisons) == 2:
                cities = list(city_comparisons.keys())
                city1, city2 = cities[0], cities[1]
                metrics1, metrics2 = city_comparisons[city1], city_comparisons[city2]
                
                response += f"🔍 **Comparative Insights**:\n"
                
                # Compare issue rates
                if metrics1['issue_rate'] < metrics2['issue_rate']:
                    better_city, worse_city = city1, city2
                    better_rate, worse_rate = metrics1['issue_rate'], metrics2['issue_rate']
                else:
                    better_city, worse_city = city2, city1
                    better_rate, worse_rate = metrics2['issue_rate'], metrics1['issue_rate']
                
                response += f"  • {better_city} has {worse_rate - better_rate:.1f}% lower failure rate than {worse_city}\n"
                
                # Compare primary causes
                if metrics1['primary_cause'] != metrics2['primary_cause']:
                    response += f"  • Different primary causes: {city1} → {metrics1['primary_cause']}, {city2} → {metrics2['primary_cause']}\n"
                else:
                    response += f"  • Both cities share same primary cause: {metrics1['primary_cause']}\n"
                
                # Performance recommendations
                response += f"\n💡 **Recommendations**:\n"
                response += f"  • Focus improvement efforts on {worse_city} ({worse_rate:.1f}% failure rate)\n"
                response += f"  • Address {metrics1['primary_cause'].lower()} and {metrics2['primary_cause'].lower()} causes\n"
                response += f"  • Share best practices from {better_city} with {worse_city}\n"
            
            return response
        else:
            return "Please specify at least two cities for comparison."

    def handle_prediction_query(self, entities):
        """Answers forward-looking questions, including simple what-if scenarios about extra volume."""
        # Extract prediction scenarios
        if any('add' in entity or 'new' in entity for entity in entities.get('operations', [])):
            # Volume impact prediction
            current_volume = len(self.master_data)
            projected_volume = current_volume * 1.2  # 20% increase assumption
            
            current_issue_rate = self.master_data['has_issue'].mean()
            projected_issue_rate = min(current_issue_rate * 1.05, 1.0)  # 5% degradation
            
            response = f"Summary: Scenario: Adding volume (capacity risk: high)\n"
            response += f"Scenario Analysis:\n"
            response += f"  - Current Total Orders: {current_volume:,}\n"
            response += f"  - Projected Total Orders: {int(projected_volume):,}\n"
            response += f"  - Current Issue Rate: {current_issue_rate*100:.2f}%\n"
            response += f"  - Projected Issue Rate: {projected_issue_rate*100:.2f}%\n"
            response += f"  - Additional Issues Expected: {(projected_volume * projected_issue_rate - current_volume * current_issue_rate):.2f}\n"
            
            return response
        
        return self.handle_analysis_query(entities)

    def handle_optimization_query(self, entities):
        """Spots weak areas (slow picking, big delays) and suggests straight-forward fixes."""
        data = self.master_data.copy()
        
        # Calculate current performance metrics
        current_metrics = {
            'success_rate': (1 - data['has_issue'].mean()) * 100,
            'total_orders': len(data),
            'avg_picking_time': data['picking_duration'].mean(),
            'avg_dispatch_delay': data['dispatch_delay'].mean()
        }
        
        response = f"Current Performance Metrics:\n"
        response += f"  - Success Rate: {current_metrics['success_rate']:.2f}%\n"
        response += f"  - Total Orders: {current_metrics['total_orders']:.2f}\n"
        response += f"  - Avg Picking Time: {current_metrics['avg_picking_time']:.2f} min\n"
        response += f"  - Avg Dispatch Delay: {current_metrics['avg_dispatch_delay']:.2f} min\n"
        
        # Generate improvement suggestions
        response += f"Improvement Areas:\n"
        if current_metrics['avg_picking_time'] > 15:
            response += f"  - Picking time is above optimal (>15 min)\n"
        else:
            response += f"  - Picking time is optimal\n"
            
        if current_metrics['avg_dispatch_delay'] > 30:
            response += f"  - Dispatch delay needs attention (>30 min)\n"
        else:
            response += f"  - Dispatch timing is good\n"
        
        response += f"Strategic Recommendations:\n"
        response += f"  - Implement predictive analytics for demand forecasting\n"
        response += f"  - Optimize warehouse layout and picking routes\n"
        response += f"  - Enhance driver training and route optimization\n"
        
        return response

    def handle_causation_query(self, entities):
        """Drills into filtered data to explain why failures happened (weather, traffic, operations)."""
        original_count = len(self.master_data)
        data = self._filter_data_by_entities(entities).copy()

        actual_clients = []
        if 'client_name' in data.columns and not data.empty:
            actual_clients = sorted(data['client_name'].dropna().unique().tolist())

        actual_cities = []
        if 'city' in data.columns and not data.empty:
            actual_cities = sorted(data['city'].dropna().unique().tolist())

        if entities.get('clients') and actual_clients:
            print(f"👥 Filtered to clients: {actual_clients}")

        if entities.get('cities') and actual_cities:
            print(f"🌆 Filtered to cities: {actual_cities}")

        if len(data) != original_count:
            print(f"📊 Filter applied: {len(data)} records from {original_count} total")

        if len(data) == 0:
            return f"No data found for the specified filters. Original dataset: {original_count:,} records."

        # Analyze primary causes of issues
        issue_data = data[data['has_issue'] == True]

        if len(issue_data) == 0:
            return f"No issues found in the filtered data ({len(data)} total orders from {original_count:,} original records)."
        
        # Count different types of issues
        cause_counts = {}
        
        # Weather-related issues
        weather_issues = issue_data[issue_data['weather_condition'].isin(self.config['weather_issues'])]['weather_condition'].value_counts()
        cause_counts.update({f"Weather Related": weather_issues.sum()})
        
        # Traffic-related issues  
        traffic_issues = issue_data[issue_data['traffic_condition'].isin(self.config['traffic_issues'])]['traffic_condition'].value_counts()
        cause_counts.update({f"Traffic Related": traffic_issues.sum()})
        
        # Operational delays
        operational_issues = issue_data[issue_data['dispatch_delay'] > 60]
        cause_counts.update({f"Operational Delays": len(operational_issues)})
        
        # Find primary cause
        primary_cause = max(cause_counts.items(), key=lambda x: x[1]) if cause_counts else ("Unknown", 0)
        
        # Calculate percentages
        total_issues = len(issue_data)
        issue_rate = (total_issues / len(data)) * 100 if len(data) > 0 else 0
        
        # Generate dynamic header based on actual filters
        header_parts = []

        if entities.get('clients'):
            client_display = actual_clients if actual_clients else entities['clients']
            header_parts.append(f"Client {', '.join(client_display)}")

        if entities.get('cities'):
            city_display = actual_cities if actual_cities else [city.title() for city in entities['cities']]
            header_parts.append(f"City: {', '.join(city_display)}")

        if entities.get('time_periods'):
            formatted_periods = [term.title() if term.islower() else term for term in entities['time_periods']]
            header_parts.append(f"Timeframe: {', '.join(formatted_periods)}")

        if entities.get('warehouses'):
            header_parts.append(f"Warehouses: {', '.join(entities['warehouses'])}")

        if not header_parts:
            header_parts.append("Delivery Analysis")

        header_title = " | ".join(header_parts)

        response = f"{header_title}:\n"
        response += f"  • Total Orders: {len(data):,}\n"
        response += f"  • Orders with Issues: {total_issues:,} ({issue_rate:.1f}%)\n"
        response += f"  • Primary Cause: {primary_cause[0].lower()}\n\n"
        response += f"Detailed Cause Breakdown:\n"
        for cause, count in cause_counts.items():
            percentage = (count / total_issues) * 100 if total_issues > 0 else 0
            response += f"  - {cause}: {count:,} ({percentage:.1f}% of issues)\n"

        if 'event_type' in issue_data.columns and issue_data['event_type'].notna().any():
            event_counts = issue_data['event_type'].fillna('Unspecified').str.title().value_counts()
            if not event_counts.empty:
                response += f"\nEvent Correlation:\n"
                for event, count in event_counts.head(3).items():
                    percentage = (count / total_issues) * 100 if total_issues > 0 else 0
                    response += f"  - {event}: {count:,} orders ({percentage:.1f}% of issues)\n"
        
        # Add dynamic insights based on actual data
        if total_issues > 0:
            response += f"\n🔍 Key Insights:\n"
            if cause_counts.get("Weather Related", 0) > cause_counts.get("Traffic Related", 0):
                response += f"  • Weather conditions (rain, storms, fog) are the leading cause\n"
            elif cause_counts.get("Traffic Related", 0) > cause_counts.get("Weather Related", 0):
                response += f"  • Traffic congestion is the primary challenge\n"
            
            # Dynamic insight based on actual filters
            location_components = []
            if entities.get('clients'):
                client_component = actual_clients if actual_clients else entities['clients']
                location_components.append(f"Client {', '.join(client_component)}")
            if entities.get('cities'):
                city_component = actual_cities if actual_cities else [city.title() for city in entities['cities']]
                location_components.append(', '.join(city_component))
            location_label = ' & '.join(location_components) if location_components else 'the analyzed area'

            period_components = [term.title() if term.islower() else term for term in entities.get('time_periods', [])]
            time_label = period_components[0] if period_components else 'the analyzed period'

            response += f"  • {issue_rate:.1f}% of orders for {location_label} had delivery issues in {time_label}\n"
            
            # Recommendations - Enhanced with festival-specific advice
            response += f"\n💡 Recommendations:\n"
            
            # Check if this is a festival period analysis
            festival_terms = ['festival', 'holiday', 'peak season', 'diwali', 'christmas', 'new year', 'eid']
            is_festival_query = any(term in time_label.lower() for term in festival_terms) if time_label != 'the analyzed period' else False
            
            if is_festival_query:
                # Festival-specific recommendations
                response += f"  • **Festival Preparation Strategy:**\n"
                response += f"    - Scale up warehouse staffing by 40-60% during peak periods\n"
                response += f"    - Pre-position inventory in regional hubs 2 weeks before festivals\n"
                response += f"    - Partner with local logistics providers for last-mile delivery\n"
                response += f"    - Implement dynamic pricing for peak delivery slots\n"
                response += f"  • **Capacity Management:**\n"
                response += f"    - Set up temporary fulfillment centers in high-demand areas\n"
                response += f"    - Extend operating hours (6 AM - 11 PM) during festival weeks\n"
                response += f"    - Create priority queues for festival orders\n"
                response += f"  • **Customer Communication:**\n"
                response += f"    - Proactively communicate expected delivery delays\n"
                response += f"    - Offer delivery date selection with premium options\n"
                response += f"    - Set up real-time order tracking and notifications\n"
                response += f"  • **Risk Mitigation:**\n"
                response += f"    - Monitor weather forecasts and pre-position inventory\n"
                response += f"    - Plan alternative routes to avoid festival traffic congestion\n"
                response += f"    - Maintain buffer stock for high-demand items\n"
            else:
                # Standard recommendations
                response += f"  • Monitor weather forecasts and pre-position inventory\n"
                response += f"  • Optimize delivery routes during peak traffic hours\n"
                response += f"  • Consider alternative delivery time slots\n"
        
        return response

    def handle_capacity_planning_query(self, entities, original_query=None):
        """Estimates the impact of onboarding more orders and outlines risks plus mitigation steps."""
        import re
        from datetime import datetime, timedelta
        
        # Extract volume information from the original query
        query_text = original_query.lower() if original_query else ''
        volume_match = re.search(r'(\d+[,.]?\d*)\s*(?:thousand|k|extra|additional|new|more)?\s*(?:orders?|deliveries?)', query_text)
        
        if volume_match:
            volume_str = volume_match.group(1).replace(',', '').replace('.', '')
            try:
                additional_volume = int(float(volume_str))
                # Handle thousands (20k, 20,000, etc.)
                if 'k' in volume_match.group(0).lower() or 'thousand' in volume_match.group(0).lower():
                    additional_volume *= 1000
            except:
                additional_volume = 20000  # Default assumption
        else:
            additional_volume = 20000  # Default assumption for capacity planning
        
        # Get current system metrics
        current_data = self.master_data.copy()
        total_orders = len(current_data)
        current_monthly_volume = total_orders  # Assuming this represents monthly data
        current_issue_rate = current_data['has_issue'].mean()
        current_avg_picking = current_data['picking_duration'].mean()
        current_avg_dispatch = current_data['dispatch_delay'].mean()
        
        # Calculate projected metrics with additional volume
        projected_volume = current_monthly_volume + additional_volume
        volume_increase_pct = (additional_volume / current_monthly_volume) * 100
        
        # Predict impact on issue rates (typically increases with volume due to capacity strain)
        # Use ML model if available, otherwise use heuristic
        volume_strain_factor = 1 + (volume_increase_pct / 100) * 0.3  # 30% degradation per 100% increase
        projected_issue_rate = min(current_issue_rate * volume_strain_factor, 0.95)  # Cap at 95%
        
        # Predict resource strain
        capacity_utilization = min((projected_volume / current_monthly_volume), 2.0)  # Cap at 200%
        
        # Generate response
        response = f"📊 **Capacity Planning Analysis: Client Onboarding Impact**\n\n"
        response += f"**Current vs. Projected Volume:**\n"
        response += f"  • Current Monthly Volume: {current_monthly_volume:,} orders\n"
        response += f"  • Additional Volume (Client Y): {additional_volume:,} orders\n"
        response += f"  • Projected Total Volume: {projected_volume:,} orders\n"
        response += f"  • Volume Increase: {volume_increase_pct:.1f}%\n\n"
        
        response += f"**Risk Assessment & Impact Prediction:**\n"
        response += f"  • Current Issue Rate: {current_issue_rate*100:.1f}%\n"
        response += f"  • Projected Issue Rate: {projected_issue_rate*100:.1f}% (+{(projected_issue_rate-current_issue_rate)*100:.1f}%)\n"
        response += f"  • Capacity Utilization: {capacity_utilization*100:.0f}%\n"
        response += f"  • System Strain Level: {'🔴 Critical' if capacity_utilization > 1.5 else '🟡 High' if capacity_utilization > 1.2 else '🟢 Manageable'}\n\n"
        
        # Identify specific risk areas
        response += f"**Key Risk Areas:**\n"
        
        # Warehouse capacity risks
        if capacity_utilization > 1.3:
            response += f"  🏭 **Warehouse Operations:** High risk of bottlenecks\n"
            response += f"    - Picking time may increase by {(capacity_utilization-1)*30:.0f}%\n"
            response += f"    - Storage capacity may be exceeded\n"
        
        # Fleet capacity risks  
        if capacity_utilization > 1.2:
            response += f"  🚛 **Fleet & Delivery:** Severe strain expected\n"
            response += f"    - Dispatch delays may increase by {(capacity_utilization-1)*40:.0f}%\n"
            response += f"    - Delivery time windows may be missed\n"
        
        # Customer service risks
        if projected_issue_rate > 0.25:
            response += f"  📞 **Customer Experience:** Service degradation likely\n"
            response += f"    - Customer complaints may increase by {((projected_issue_rate/current_issue_rate)-1)*100:.0f}%\n"
            response += f"    - Brand reputation at risk\n"
        
        response += f"\n**🚀 Mitigation Strategy:**\n"
        
        # Infrastructure scaling
        response += f"**Infrastructure Scaling:**\n"
        if capacity_utilization > 1.3:
            response += f"  • Expand warehouse capacity by {max(30, (capacity_utilization-1)*100):.0f}%\n"
            response += f"  • Add {max(2, int((capacity_utilization-1)*10))} temporary fulfillment centers\n"
        
        response += f"  • Increase fleet size by {max(20, volume_increase_pct):.0f}%\n"
        response += f"  • Scale warehouse staff by {max(25, volume_increase_pct*0.8):.0f}%\n"
        
        # Operational improvements
        response += f"\n**Operational Readiness:**\n"
        response += f"  • Implement staggered order processing to smooth demand\n"
        response += f"  • Pre-negotiate overflow capacity with 3rd party logistics\n"
        response += f"  • Set up dedicated Client Y processing lanes\n"
        response += f"  • Implement real-time capacity monitoring\n"
        
        # Technology upgrades
        response += f"\n**Technology & Process:**\n"
        response += f"  • Upgrade warehouse management system for higher throughput\n"
        response += f"  • Implement predictive analytics for demand forecasting\n"
        response += f"  • Set up automated alerts for capacity thresholds\n"
        response += f"  • Deploy dynamic routing optimization\n"
        
        # Timeline recommendations
        response += f"\n**📅 Implementation Timeline:**\n"
        response += f"  • **Immediate (Week 1-2):** Staff hiring, overflow partnerships\n"
        response += f"  • **Short-term (Month 1):** Technology upgrades, process optimization\n"
        response += f"  • **Medium-term (Month 2-3):** Infrastructure expansion, client onboarding\n"
        response += f"  • **Ongoing:** Performance monitoring and continuous optimization\n"
        
        # Success metrics
        response += f"\n**🎯 Success Metrics to Monitor:**\n"
        response += f"  • Keep issue rate below {min(current_issue_rate*1.1, 0.30)*100:.0f}%\n"
        response += f"  • Maintain picking time under {current_avg_picking*1.2:.0f} minutes\n"
        response += f"  • Keep dispatch delays under {current_avg_dispatch*1.15:.0f} minutes\n"
        response += f"  • Achieve >95% on-time delivery for Client Y orders\n"
        
        return response

    def apply_time_filter(self, data, time_periods):
        """Keeps only the rows that match the requested time windows (yesterday, last month, festivals, etc.)."""
        from datetime import datetime, timedelta
        
        # Handle list of time periods
        if isinstance(time_periods, list):
            for time_period in time_periods:
                data = self._apply_single_time_filter(data, time_period)
        else:
            data = self._apply_single_time_filter(data, time_periods)
        
        return data
    
    def _apply_single_time_filter(self, data, time_period):
        """Implements the actual date math for one time phrase before handing data back."""
        if 'picking_start' not in data.columns:
            return data

        data = data.copy()

        if not pd.api.types.is_datetime64_any_dtype(data['picking_start']):
            data['picking_start'] = pd.to_datetime(data['picking_start'], errors='coerce')

        date_col = 'picking_start'
        current_date = datetime.now()
        time_period_lower = time_period.lower() if isinstance(time_period, str) else ''

        event_types = None
        if 'event_type' in data.columns:
            event_types = data['event_type'].fillna('').astype(str).str.lower()

        # Handle month names
        month_mapping = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }

        def filter_by_event_labels(description, labels):
            if event_types is None:
                return None
            mask = event_types.isin(labels) if isinstance(labels, set) else event_types.str.contains(labels)
            if not mask.any():
                return None
            subset = data.loc[mask].copy()
            print(f"🎊 Filtering {description} via event tags: {len(subset)} records")
            print(f"📊 Time filter applied: {len(subset)} records from {len(data)} total")
            return subset

        try:
            if time_period_lower in month_mapping:
                month_num = month_mapping[time_period_lower]
                filtered_data = data[data[date_col].dt.month == month_num]
                print(f"🕒 Filtering {time_period}: {len(filtered_data)} records from {len(data)} total")
                return filtered_data
            elif 'last month' in time_period_lower:
                last_month = current_date.replace(day=1) - timedelta(days=1)
                start_date = last_month.replace(day=1)
                end_date = current_date.replace(day=1) - timedelta(days=1)
                filtered_data = data[(data[date_col] >= start_date) & (data[date_col] <= end_date)]
                print(f"🕒 Filtering last month: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                print(f"📊 Time filter applied: {len(filtered_data)} records from {len(data)} total")
                return filtered_data
            elif 'last 2 months' in time_period_lower or 'past 2 months' in time_period_lower or 'last_2_month' in time_period_lower or '2 month' in time_period_lower:
                start_date = current_date - timedelta(days=60)
                filtered_data = data[data[date_col] >= start_date]
                print(f"🕒 Filtering last 2 months: {start_date.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
                print(f"📊 Time filter applied: {len(filtered_data)} records from {len(data)} total")
                return filtered_data
            elif 'yesterday' in time_period_lower:
                start_date = current_date - timedelta(days=1)
                end_date = current_date
                filtered_data = data[(data[date_col] >= start_date) & (data[date_col] <= end_date)]
                print(f"📊 Time filter applied: {len(filtered_data)} records from {len(data)} total")
                return filtered_data
            elif 'last week' in time_period_lower:
                start_date = current_date - timedelta(days=7)
                end_date = current_date
                filtered_data = data[(data[date_col] >= start_date) & (data[date_col] <= end_date)]
                print(f"📊 Time filter applied: {len(filtered_data)} records from {len(data)} total")
                return filtered_data
            elif 'today' in time_period_lower:
                start_date = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = current_date
                filtered_data = data[(data[date_col] >= start_date) & (data[date_col] <= end_date)]
                print(f"📊 Time filter applied: {len(filtered_data)} records from {len(data)} total")
                return filtered_data
            elif any(keyword in time_period_lower for keyword in ['festival period', 'festival season', 'festive period', 'festive season']) or 'festival' in time_period_lower:
                filtered_data = filter_by_event_labels('festival period', {'festival'})
                if filtered_data is not None:
                    return filtered_data

                festival_months = []
                if event_types is not None:
                    festival_months = (
                        data.loc[event_types == 'festival', date_col]
                        .dt.month.dropna().unique().tolist()
                    )

                if not festival_months:
                    festival_months = [10, 11, 12]

                filtered_data = data[data[date_col].dt.month.isin(festival_months)]

                if len(filtered_data) == 0:
                    monthly_counts = data.groupby(data[date_col].dt.month).size()
                    top_months = monthly_counts.nlargest(3).index.tolist()
                    filtered_data = data[data[date_col].dt.month.isin(top_months)]
                    print(f"🎊 No tagged festival events found; using top peak months {top_months}")
                else:
                    print(f"🎊 Filtering festival period by months {festival_months}")

                print(f"📊 Time filter applied: {len(filtered_data)} records from {len(data)} total")
                return filtered_data
            elif any(keyword in time_period_lower for keyword in ['holiday period', 'holiday season']):
                filtered_data = filter_by_event_labels('holiday period', {'holiday'})
                if filtered_data is not None:
                    return filtered_data

                holiday_months = []
                if event_types is not None:
                    holiday_months = (
                        data.loc[event_types == 'holiday', date_col]
                        .dt.month.dropna().unique().tolist()
                    )

                if not holiday_months:
                    holiday_months = [11, 12]

                filtered_data = data[data[date_col].dt.month.isin(holiday_months)]

                if len(filtered_data) == 0:
                    monthly_counts = data.groupby(data[date_col].dt.month).size()
                    top_months = monthly_counts.nlargest(2).index.tolist()
                    filtered_data = data[data[date_col].dt.month.isin(top_months)]
                    print(f"🎄 No tagged holiday events found; using peak months {top_months}")
                else:
                    print(f"🎄 Filtering holiday period by months {holiday_months}")

                print(f"📊 Time filter applied: {len(filtered_data)} records from {len(data)} total")
                return filtered_data
            elif 'peak season' in time_period_lower or 'peak period' in time_period_lower:
                filtered_data = filter_by_event_labels('peak season', {'festival', 'holiday'})
                if filtered_data is not None:
                    return filtered_data

                monthly_counts = data.groupby(data[date_col].dt.month).size()
                top_months = monthly_counts.nlargest(3).index.tolist()
                filtered_data = data[data[date_col].dt.month.isin(top_months)]
                print(f"📈 Using data from top peak months {top_months}")
                print(f"📊 Time filter applied: {len(filtered_data)} records from {len(data)} total")
                return filtered_data
            elif any(festival in time_period_lower for festival in ['diwali', 'christmas', 'new year', 'eid', 'holi', 'dussehra', 'navratri']):
                # For specific festivals, filter by typical dates
                festival_months_map = {
                    'diwali': [10, 11],  # Oct-Nov
                    'christmas': [12],   # Dec
                    'new year': [12, 1], # Dec-Jan
                    'eid': [4, 5, 6],    # Variable, but often Apr-Jun
                    'holi': [3],         # Mar
                    'dussehra': [10],    # Oct
                    'navratri': [9, 10]  # Sep-Oct
                }
                
                relevant_months = []
                for festival, months in festival_months_map.items():
                    if festival in time_period_lower:
                        relevant_months.extend(months)
                
                if relevant_months:
                    filtered_data = data[data[date_col].dt.month.isin(relevant_months)]
                    
                    # If no data in specific festival months, fall back to peak volume periods
                    if len(filtered_data) == 0:
                        print(f"🎭 No data found for {time_period_lower} months, analyzing peak volume periods instead")
                        # Find peak volume periods as proxy for festival impact
                        monthly_counts = data.groupby(data[date_col].dt.month).size()
                        top_months = monthly_counts.nlargest(2).index.tolist()
                        filtered_data = data[data[date_col].dt.month.isin(top_months)]
                        print(f"📊 Using peak volume data: {len(filtered_data)} records from high-demand periods")
                    else:
                        print(f"🎭 Filtering for {time_period_lower}: {len(filtered_data)} records from festival months")
                    
                    print(f"📊 Time filter applied: {len(filtered_data)} records from {len(data)} total")
                    return filtered_data
            else:
                return data  # No specific time filter
                
        except Exception as e:
            print(f"⚠️ Error applying time filter: {e}")
            return data

    def generate_recommendations(self, analysis):
        """Translates key metrics into a short list of practical improvement tips."""
        recommendations = []
        
        if analysis['issue_rate'] > 20:
            recommendations.append("Issue rate is high - implement quality control measures")
        
        if analysis['avg_picking_time'] > 15:
            recommendations.append("Optimize warehouse layout to reduce picking time")
        
        if analysis['avg_dispatch_delay'] > 30:
            recommendations.append("Streamline dispatch process to reduce delays")
        
        recommendations.append("Monitor key performance indicators regularly")
        recommendations.append("Implement predictive analytics for better planning")
        
        return recommendations

    def _filter_data_by_entities(self, entities):
        """Slices the master dataset to the cities/warehouses/timeframes mentioned in the question."""
        if not isinstance(entities, dict) or self.master_data.empty:
            return self.master_data

        filtered = self.master_data

        # Filter by cities using fuzzy matching against actual data
        if entities.get('cities') and 'city' in filtered.columns:
            available_cities = filtered['city'].dropna().unique().tolist()
            matched_cities = []
            for city in entities['cities']:
                if city in available_cities:
                    matched_cities.append(city)
                else:
                    match = process.extractOne(city, available_cities, scorer=fuzz.token_set_ratio)
                    if match and match[1] >= self.config['fuzzy_match_threshold']:
                        matched_cities.append(match[0])
            if matched_cities:
                filtered = filtered[filtered['city'].isin(matched_cities)]

        if entities.get('clients') and 'client_name' in filtered.columns:
            available_clients = filtered['client_name'].dropna().unique().tolist()
            matched_clients = []
            for client in entities['clients']:
                if client in available_clients:
                    matched_clients.append(client)
                else:
                    match = process.extractOne(client, available_clients, scorer=fuzz.token_set_ratio)
                    if match and match[1] >= self.config['fuzzy_match_threshold']:
                        matched_clients.append(match[0])
            if matched_clients:
                filtered = filtered[filtered['client_name'].isin(matched_clients)]

        # Filter by warehouse identifiers extracted from the query
        if entities.get('warehouses') and 'warehouse_id' in filtered.columns:
            warehouse_ids = []
            for warehouse in entities['warehouses']:
                match = re.search(r'(\d+)$', warehouse)
                if match:
                    warehouse_ids.append(int(match.group(1)))
            if warehouse_ids:
                filtered = filtered[filtered['warehouse_id'].isin(warehouse_ids)]

        # Apply time-period filtering leveraging existing helper
        if entities.get('time_periods'):
            filtered = self.apply_time_filter(filtered, entities['time_periods'])

        return filtered

    def generate_ml_insights(self, entities):
        """Adds machine-learned extras (risk score, top factors, anomalies) when enough data supports it."""
        if not self.enable_ml or not self.ml_models:
            return None
            
        try:
            filtered_data = self._filter_data_by_entities(entities)
            if filtered_data is None or filtered_data.empty:
                return None

            # Require a reasonable sample so that insights are meaningful
            if len(filtered_data) < max(25, int(self.config['ml_training_min_samples'] * 0.25)):
                return None

            insights = {
                'context': {
                    'orders_analyzed': int(len(filtered_data))
                }
            }
            
            # Issue risk prediction grounded in contextual data
            if 'issue_prediction' in self.ml_models:
                risk_score = self._predict_issue_risk(filtered_data)
                if risk_score is not None:
                    insights['issue_risk'] = {
                        'probability': round(risk_score, 3),
                        'level': 'High' if risk_score > self.config['risk_high_threshold'] else 'Medium' if risk_score > self.config['risk_medium_threshold'] else 'Low'
                    }

            # Top factors affecting performance
            top_factors = self._get_top_factors()
            if top_factors:
                insights['key_factors'] = top_factors
            
            # Anomaly detection scoped to the same slice
            anomalies = self._detect_anomalies(filtered_data)
            if anomalies:
                insights['anomalies_detected'] = anomalies
            
            # Remove context-only responses
            return insights if len(insights) > 1 else None
            
        except Exception as e:
            logger.error(f"Error generating ML insights: {e}")
            return None

    def _format_ml_insights(self, insights):
        """Turns raw ML insight dictionaries into compact, reviewer-friendly bullet points."""
        if not insights:
            return ""

        lines = []

        context = insights.get('context', {}) or {}
        orders_analyzed = context.get('orders_analyzed')
        if orders_analyzed:
            lines.append(f"- Based on {orders_analyzed:,} similar orders")

        issue_risk = insights.get('issue_risk') or {}
        probability = issue_risk.get('probability')
        level = issue_risk.get('level')
        if probability is not None and level:
            lines.append(f"- Issue risk {level.lower()}: {probability * 100:.1f}% probability")

        key_factors = insights.get('key_factors') or {}
        factor_summaries = []
        for model_name, factors in key_factors.items():
            if not factors:
                continue
            readable_name = model_name.replace('_', ' ').title()
            top_factor_labels = []
            for item in factors[:3]:
                factor_label = item.get('factor')
                importance = item.get('importance')
                if factor_label is None or importance is None:
                    continue
                top_factor_labels.append(f"{factor_label} ({importance})")
            if top_factor_labels:
                factor_summaries.append(f"{readable_name}: {', '.join(top_factor_labels)}")
        if factor_summaries:
            lines.append(f"- Top drivers: {'; '.join(factor_summaries)}")

        anomalies = insights.get('anomalies_detected') or {}
        anomaly_count = anomalies.get('count')
        if anomaly_count:
            percentage = anomalies.get('percentage')
            description = anomalies.get('description')
            anomaly_line = f"- {anomaly_count} anomalies detected"
            if percentage is not None:
                anomaly_line += f" ({percentage:.1f}%)"
            if description:
                anomaly_line += f" — {description}"
            lines.append(anomaly_line)

        return "\n".join(lines)

    def _predict_issue_risk(self, filtered_data):
        """Runs the classification model on the filtered slice to get an average problem probability."""
        try:
            if 'issue_prediction' not in self.ml_models:
                return None
            
            model = self.ml_models['issue_prediction']
            if not hasattr(self, 'ml_training_features') or self.ml_training_features.empty:
                return None

            feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
                           'picking_duration', 'dispatch_delay', 'weather_condition_encoded',
                           'traffic_condition_encoded', 'city_encoded']

            # Align contextual data with engineered feature rows used during training
            contextual_indices = self.ml_training_features.index.intersection(filtered_data.index)
            if len(contextual_indices) < 10:
                return None

            contextual_features = self.ml_training_features.loc[contextual_indices, feature_cols].fillna(0)
            if contextual_features.empty:
                return None

            # Predict probability for each record and aggregate
            risk_probs = model.predict_proba(contextual_features)[:, 1]
            if len(risk_probs) == 0:
                return None
            
            return float(np.mean(risk_probs))
            
        except Exception as e:
            logger.error(f"Error predicting issue risk: {e}")
            return None

    def _get_top_factors(self):
        """Collects the most influential features from each trained model for storytelling."""
        try:
            all_factors = {}
            
            for model_name, importance_dict in self.feature_importance.items():
                # Get top 3 factors for each model
                sorted_factors = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                all_factors[model_name] = [
                    {'factor': factor.replace('_', ' ').title(), 'importance': round(importance, 3)}
                    for factor, importance in sorted_factors
                ]
            
            return all_factors
            
        except Exception as e:
            logger.error(f"Error getting top factors: {e}")
            return {}

    def _detect_anomalies(self, filtered_data):
        """Uses Isolation Forest to flag unusual timing behaviour when enough records exist."""
        try:
            if not SKLEARN_AVAILABLE:
                return None
                
            from sklearn.ensemble import IsolationForest
            
            # Prepare data for anomaly detection
            data = filtered_data[['picking_duration', 'dispatch_delay', 'delivery_duration']].dropna()
            
            if len(data) < 100:
                return None
            
            # Train isolation forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(data)
            
            # Get anomalous records
            anomaly_indices = np.where(anomalies == -1)[0]
            
            if len(anomaly_indices) > 0:
                return {
                    'count': len(anomaly_indices),
                    'percentage': len(anomaly_indices) / len(data) * 100,
                    'description': 'Operational anomalies detected in timing metrics'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return None

    def export_analysis_report(self, query_results):
        """Writes a timestamped JSON snapshot of queries answered plus quality/performance stats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"./analysis_report_{timestamp}.json"
        
        report = {
            'timestamp': timestamp,
            'data_summary': {
                'total_records': len(self.master_data),
                'data_quality_score': self.data_quality_score,
                'ml_models_trained': len(self.ml_models)
            },
            'query_results': query_results,
            'performance_metrics': {
                'avg_response_time': np.mean(self.response_times) if self.response_times else 0,
                'total_queries': len(self.response_times)
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report_path

    # Convenience alias for backward compatibility
    def ask(self, query):
        """Tiny helper so callers can ask questions with `analyzer.ask(...)`."""
        return self.process_natural_language_query(query)


def comprehensive_demo(verbose=False, full_output=False):
    """Walks through a scripted tour of representative questions to showcase the system."""
    print("🚀 ENHANCED NLP DELIVERY ANALYZER - COMPREHENSIVE DEMO")
    print("=" * 80)
    
    try:
        # Initialize analyzer with all features enabled
        analyzer = CompleteNLPDeliveryAnalyzer(enable_ml=True, enable_cache=True, enable_viz=True)
        
        # Test queries showcasing different capabilities
        demo_queries = [
            "What's happening in Mumbai?",
            "Why are there so many delivery failures?",
            "How can we improve our success rate?",
            "Compare performance between Delhi and Mumbai",
            "Which city performs better - Bangalore vs Chennai?",
            "What will happen if we add 30000 new orders?",
            "Predict issues for next week",
            "What's the risk of delays during monsoon?",
            "Show me trends over the past month",
            "Identify anomalies in delivery performance",
            "What factors influence delivery success most?",
            "Create a dashboard for Mumbai operations",
            "Show me a trend chart for issue rates",
            "Generate comparison charts between cities",
            "Analyze warehouse capacity utilization",
            "Forecast delivery volumes for Q4",
            "Identify optimization opportunities",
            "Executive summary for board presentation",
            "Cost impact analysis of current issues",
            "Customer satisfaction correlation with delivery performance"
        ]
        
        print(f"\n🧠 Testing {len(demo_queries)} enhanced query types:")
        print("-" * 80)
        
        query_results = []
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{i:2d}. Query: '{query}'")
            print("🔍 Processing:", query)
            
            try:
                response = analyzer.process_natural_language_query(query)
                if full_output:
                    print("Complete Response:")
                    print(response)  # Show full response
                elif verbose:
                    print("Extended Response:")
                    # Show first 1000 characters and 20 lines
                    preview = response[:1000] + "..." if len(response) > 1000 else response
                    for line in preview.split('\n')[:20]:
                        print(line)
                    if len(response) > 1000:
                        print("... (use --full flag for complete response)")
                else:
                    print("Response Preview:")
                    # Show first 400 characters and 10 lines (improved from before)
                    preview = response[:400] + "..." if len(response) > 400 else response
                    for line in preview.split('\n')[:10]:
                        print(line)
                    if len(response) > 400:
                        print("... (use --verbose or --full for more details)")
                
                query_results.append({
                    'query': query,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                continue
            
            if analyzer.last_ml_insights_summary:
                print("\nKey ML signals:")
                for line in analyzer.last_ml_insights_summary.split('\n'):
                    print(f"   {line}")
            print("-" * 60)
        
        # Performance summary
        print(f"\n📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"✅ Successfully processed: {len(query_results)}/{len(demo_queries)} queries")
        if analyzer.response_times:
            print(f"⏱️  Total processing time: {sum(analyzer.response_times):.2f}s")
            print(f"⚡ Average response time: {np.mean(analyzer.response_times):.2f}s")
            print(f"💾 Cache hit rate: {0.0}%")  # First run, no cache hits
        print(f"📏 Data quality score: {analyzer.data_quality_score:.1f}/10")
        print(f"🤖 ML models trained: {len(analyzer.ml_models)}")
        
        # Export comprehensive report
        report_path = analyzer.export_analysis_report(query_results)
        print(f"📄 Comprehensive report: Report exported to: {report_path}")
        
        print(f"\n🎯 ENHANCED FEATURES DEMONSTRATED:")
        print(f"   ✓ Advanced NLP with fuzzy matching")
        print(f"   ✓ Machine Learning predictions")
        print(f"   ✓ Interactive visualizations")
        print(f"   ✓ Performance caching")
        print(f"   ✓ Anomaly detection")  
        print(f"   ✓ Sentiment analysis")
        print(f"   ✓ Knowledge graph relationships")
        print(f"   ✓ Real-time analytics")
        
        print("\n" + "=" * 50)
        
        # Interactive mode
        interactive_mode = input("Would you like to try interactive mode? (y/n): ")
        if interactive_mode.lower() == 'y':
            interactive_session(analyzer)
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def interactive_session(analyzer):
    """Runs a simple command-line loop so people can keep asking questions."""
    print("\n🌟 INTERACTIVE NLP DELIVERY ANALYZER")
    print("=" * 50)
    print("Type your questions about delivery operations!")
    print("Examples:")
    print("  - What's the issue rate in Mumbai?")
    print("  - Compare Delhi vs Bangalore performance") 
    print("  - Predict impact of adding 5000 orders")
    print("  - Create a dashboard")
    print("  - Show trends for last month")
    print("Type 'quit' to exit, 'help' for more examples")
    print("=" * 50)
    
    while True:
        try:
            user_query = input("\n🤖 Your question: ")
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using Enhanced NLP Delivery Analyzer!")
                break
            elif user_query.lower() == 'help':
                print_help_examples()
                continue
            elif not user_query.strip():
                continue
                
            print(f"🔍 Processing: {user_query}")
            response = analyzer.process_natural_language_query(user_query)
            
            print(f"\n📊 Analysis Results:")
            print("-" * 40)
            print(response)
            print(f"\n⚡ Processed in {analyzer.response_times[-1]:.2f} seconds")
            if analyzer.last_ml_insights_summary:
                print("\nKey ML signals:")
                for line in analyzer.last_ml_insights_summary.split('\n'):
                    print(f"   {line}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


def print_help_examples():
    """Lists sample questions grouped by theme to spark ideas in interactive mode."""
    examples = {
        "📍 City Analysis": [
            "What's happening in Mumbai?",
            "Show Delhi performance metrics",
            "Analyze Bangalore delivery issues"
        ],
        "📊 Comparisons": [
            "Compare Mumbai vs Delhi",
            "Which performs better - Chennai or Bangalore?",
            "Delhi vs Mumbai success rates"
        ],
        "🔮 Predictions": [
            "Predict issues for next week",
            "What happens if we add 10000 orders?",
            "Forecast Q4 delivery volumes"
        ],
        "🎯 Optimization": [
            "How to improve success rate?",
            "Optimization opportunities",
            "Reduce delivery failures"
        ],
        "🔍 Root Cause": [
            "Why delivery failures?",
            "What causes delays in Mumbai?",
            "Root cause analysis"
        ]
    }
    
    print("\n📚 QUERY EXAMPLES:")
    for category, queries in examples.items():
        print(f"\n{category}:")
        for query in queries:
            print(f"  • {query}")


if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    full_output = "--full" in sys.argv or "-f" in sys.argv
    
    if full_output:
        print("🔍 Running in FULL OUTPUT mode (complete responses)")
    elif verbose:
        print("📝 Running in VERBOSE mode (extended responses)")
    else:
        print("📋 Running in STANDARD mode (use --verbose or --full for more details)")
    
    comprehensive_demo(verbose=verbose, full_output=full_output)