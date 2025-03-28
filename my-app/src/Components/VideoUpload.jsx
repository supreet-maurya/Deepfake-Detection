import React, { useState } from 'react';
import axios from 'axios';

function VideoUpload() {
  const [videoFile, setVideoFile] = useState(null); // State for selected file
  const [result, setResult] = useState(null); // State to store prediction results
  const [loading, setLoading] = useState(false); // State for loading indicator

  // Handler for when a video is selected
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setVideoFile(file);
      console.log("Selected video:", file);
    }
  };

  // Handle the upload and prediction
  const handleUpload = () => {
    if (videoFile) {
      const formData = new FormData();
      formData.append("file", videoFile); // Append the file with key 'file' (Flask expects this key)

      setLoading(true); // Set loading state
      setResult(null); // Reset result state

      // Upload video to backend using Axios
      axios
        .post("http://127.0.0.1:5000/predict", formData, {
          headers: {
            "Content-Type": "multipart/form-data", // Important for file uploads
          },
        })
        .then((response) => {
          console.log("Prediction result:", response.data);

          // Extract relevant data from the API response
          const { status, mean_fake_score, total_frames, total_evaluated_frames, total_fake_frames, total_real_frames } = response.data;

          // Store the result in state
          setResult({
            status,
            mean_fake_score,
            total_frames,
            total_evaluated_frames,
            total_fake_frames,
            total_real_frames,
          });
        })
        .catch((error) => {
          console.error("Error during prediction:", error);
          setResult({ error: "Failed to get prediction. Please try again later." });
        })
        .finally(() => setLoading(false));
    } else {
      alert("Please select a video first.");
    }
  };

  return (
    <div className="container p-6">
      <div className="text-xl border-2 flex justify-center mx-auto py-4 w-64 rounded-md mb-4">
        Upload Video
      </div>

      <div className="flex flex-col items-center space-y-4">
        {/* Input for video selection */}
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="border border-orange-300 rounded-md p-2"
        />

        {/* Show selected video */}
        {videoFile && <h2 className="text-sm">Selected Video: {videoFile.name}</h2>}

        {/* Upload button */}
        <button
          onClick={handleUpload}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          {loading ? "Uploading..." : "Upload Video"}
        </button>

        {/* Show prediction result */}
        {result && (
          <div className="mt-4 p-4 border rounded-md bg-gray-100">
            {result.error ? (
              <h3 className="text-red-500">{result.error}</h3>
            ) : (
              <>
                <h3 className="text-lg font-semibold">Prediction Results:</h3>
                <p><strong>Status:</strong> {result.status}</p>
                <p><strong>Mean Fake Score:</strong> {result.mean_fake_score.toFixed(2)}</p>
                <p><strong>Total Frames:</strong> {result.total_frames}</p>
                <p><strong>Total Evaluated Frames:</strong> {result.total_evaluated_frames}</p>
                <p><strong>Total Fake Frames:</strong> {result.total_fake_frames}</p>
                <p><strong>Total Real Frames:</strong> {result.total_real_frames}</p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default VideoUpload;
