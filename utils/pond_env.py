from pond_env import PondCleaningEnvFromDetections

# Add reset_from_segments to allow direct input of detected segments
def reset_from_segments(self, segments):
    self.current_state = self._convert_segments_to_state(segments)
    return self.current_state
PondCleaningEnvFromDetections.reset_from_segments = reset_from_segments
