"""
Logging utilities for System 4 triadic transformation tracking
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

class TransformationLogger:
    """
    Structured logger for tracking triadic transformation processes
    """
    
    def __init__(self, log_dir: Path = Path("logs"), scenario_name: str = "default"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.scenario_name = scenario_name
        
        # Setup standard Python logger
        self.logger = logging.getLogger(f"System4.{scenario_name}")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.log_dir / f"{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        
        # Structured data storage
        self.session_data = {
            "scenario": scenario_name,
            "start_time": datetime.utcnow().isoformat(),
            "steps": [],
            "metrics": [],
            "events": []
        }
        
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """Log a transformation step"""
        self.session_data["steps"].append(step_data)
        self.logger.info(f"Step {step_data.get('global_step', 'unknown')} completed")
        
        # Save incremental JSON log
        self._save_json_log(f"step_{step_data.get('global_step', 0):03d}")
        
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None) -> None:
        """Log a performance metric"""
        metric_entry = {
            "name": metric_name,
            "value": value,
            "step": step,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.session_data["metrics"].append(metric_entry)
        self.logger.info(f"Metric {metric_name}: {value:.4f}")
        
    def log_event(self, event_type: str, description: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a significant event"""
        event_entry = {
            "type": event_type,
            "description": description,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.session_data["events"].append(event_entry)
        self.logger.info(f"Event [{event_type}]: {description}")
        
    def log_initialization(self, initial_state: Dict[str, Any]) -> None:
        """Log scenario initialization"""
        self.session_data["initialization"] = initial_state
        self.logger.info(f"Scenario {self.scenario_name} initialized")
        self._save_json_log("init")
        
    def log_completion(self, final_state: Dict[str, Any]) -> None:
        """Log transformation completion"""
        self.session_data["completion"] = {
            "final_state": final_state,
            "end_time": datetime.utcnow().isoformat(),
            "total_steps": len(self.session_data["steps"])
        }
        self.logger.info(f"Transformation complete after {len(self.session_data['steps'])} steps")
        self._save_json_log("complete")
        
    def get_step_summary(self, step_num: int) -> Optional[Dict[str, Any]]:
        """Get summary for a specific step"""
        if 0 <= step_num < len(self.session_data["steps"]):
            step = self.session_data["steps"][step_num]
            return {
                "step": step_num,
                "relevance": step.get("relevance_score", 0),
                "alignment": step.get("performance", {}).get("alignment", 0),
                "cycle": step.get("cycle", 0),
                "phase": step.get("phase", 0)
            }
        return None
    
    def get_performance_trend(self) -> List[Dict[str, float]]:
        """Get performance metrics trend over time"""
        trend = []
        for step in self.session_data["steps"]:
            if "performance" in step:
                trend.append({
                    "step": step.get("global_step", 0),
                    "alignment": step["performance"].get("alignment", 0),
                    "precision": step["performance"].get("precision", 0),
                    "accuracy": step["performance"].get("accuracy", 0)
                })
        return trend
    
    def _save_json_log(self, suffix: str) -> None:
        """Save current session data to JSON file"""
        json_file = self.log_dir / f"{self.scenario_name}_{suffix}.json"
        with open(json_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
            
    def generate_report(self) -> str:
        """Generate a human-readable report of the transformation"""
        report = []
        report.append("=" * 60)
        report.append(f"System 4 Transformation Report: {self.scenario_name}")
        report.append("=" * 60)
        
        if "initialization" in self.session_data:
            report.append(f"\nInitialized: {self.session_data['start_time']}")
            
        if self.session_data["steps"]:
            report.append(f"\nTotal Steps: {len(self.session_data['steps'])}")
            
            # Performance summary
            perf_trend = self.get_performance_trend()
            if perf_trend:
                final_perf = perf_trend[-1]
                report.append(f"\nFinal Performance:")
                report.append(f"  Alignment: {final_perf['alignment']:.4f}")
                report.append(f"  Precision: {final_perf['precision']:.4f}")
                report.append(f"  Accuracy: {final_perf['accuracy']:.4f}")
                
        if self.session_data["events"]:
            report.append(f"\nSignificant Events: {len(self.session_data['events'])}")
            for event in self.session_data["events"][-5:]:  # Last 5 events
                report.append(f"  - [{event['type']}] {event['description']}")
                
        if "completion" in self.session_data:
            report.append(f"\nCompleted: {self.session_data['completion']['end_time']}")
            
        report.append("=" * 60)
        
        return "\n".join(report)