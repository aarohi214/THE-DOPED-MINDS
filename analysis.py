import numpy as np

class DefectAnalyzer:
    def __init__(self):
        self.total_processed = 0
        self.defects_found = 0
        self.defect_types = {}

    def analyze_batch_results(self, batch_results):
        """
        Aggregates results from a batch of detections.
        """
        batch_summary = {
            "count": len(batch_results),
            "defects": 0,
            "details": []
        }
        
        for res in batch_results:
            self.total_processed += 1
            if res['has_defect']:
                self.defects_found += 1
                batch_summary["defects"] += 1
                dtype = res.get('defect_type', 'unknown')
                self.defect_types[dtype] = self.defect_types.get(dtype, 0) + 1
            
            batch_summary["details"].append(res)
            
        return batch_summary

    def get_global_stats(self):
        return {
            "total_images": self.total_processed,
            "total_defects": self.defects_found,
            "defect_rate": (self.defects_found / self.total_processed) if self.total_processed > 0 else 0,
            "defect_distribution": self.defect_types
        }
