"use client"

import React, { useState } from 'react'
import InputBox from '@/components/inputbox'
import Button from '@/components/button'
import VisualizedImage from '@/components/Visualized_Data.png'
import Image from 'next/image'

export default function Page() {
  const [learningRate, setLearningRate] = useState('')
  const [epochs, setEpochs] = useState('')

  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [predictions, setPredictions] = useState<{ "# Date": string, Predicted_Receipt_Count: number }[]>([])
  const [mse, setMSE] = useState(0)

  const [loading, setLoading] = useState(false);
  const [dataFetched, setFetched] = useState(false);

  const fetchImage = async () => {
    setLoading(true);

    const parsedLR = parseFloat(learningRate);
    const parsedEpochs = parseInt(epochs);

    const lr = !isNaN(parsedLR) && isFinite(parsedLR) ? parsedLR : 0.01;
    const eps = !isNaN(parsedEpochs) && isFinite(parsedEpochs) ? parsedEpochs : 5000;

    try {
      const response = await fetch(`http://localhost:8080/api/get-image`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ learningRate: lr, epochs: eps })
      });

      const data = await response.json();
      if (data.imageUrl) {
        setImageUrl(data.imageUrl);
        setPredictions(data.predictions);
        setMSE(data.mse);

        setFetched(true);
      } else {
        console.error("Image URL not received from server.");
      }
    } catch (error) {
      console.error("Failed to fetch image", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8">
      <div>
        <h1 className="text-3xl font-bold">Fetch Rewards Prediction</h1>
        <h2 className="text-xl mt-2">Taran Polavarapu&apos;s Project</h2>
        <p className="text-sm text-gray-600 mt-2 max-w-prose">
          During the initial visualization, I noticed that the data followed a strong linear trend, with only 366 data points available. 
         This model predicts the number of receipts for the next 12 months based on the previous 12 months of data using linear regression as it fits the best with the observed data.
          The learning rate and number of epochs are set to default values, 
          but you're free to adjust them to observe how it affects predictions. Thanks!
        </p>
      </div>

      {!dataFetched && !loading && (
        <Image 
          src={VisualizedImage}
          alt="Initial Visualization" 
          width={800} 
          height={800}
          priority 
          className="max-w-full rounded shadow-md"
        />
      )}

        <div className="flex flex-col lg:flex-row items-start gap-6 mt-6">
          {imageUrl && dataFetched && (
            <div className="w-full lg:w-3/5">
              <Image 
                src={imageUrl} 
                alt="Fetched result"
                width={800} 
                height={800}
                className="w-full h-auto max-w-full rounded shadow-md"
              />
            </div>
          )}

          {predictions.length > 0 && (
            <div className="w-full lg:w-2/5 max-w-[400px]">
              <h3 className="text-xl font-semibold mb-2">Future Predictions</h3>
              <div className="overflow-auto max-h-[400px] border border-gray-200 rounded shadow-sm">
                <table className="w-full text-xs border-collapse">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="border border-gray-300 px-2 py-1 text-left">Date</th>
                      <th className="border border-gray-300 px-2 py-1 text-left">Predicted</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictions.map((row, index) => (
                      <tr key={index} className="odd:bg-white even:bg-gray-50">
                        <td className="border border-gray-300 px-2 py-1">{row['# Date']}</td>
                        <td className="border border-gray-300 px-2 py-1">{Math.round(row['Predicted_Receipt_Count'])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        <div className="p-2">
            <p><strong>Mean Squared Error (MSE):</strong> {mse}</p>
         </div>


  
      <div className="p-2">
        <InputBox
          placeholder="Learning Rate: 0.01"
          onChange={(value) => setLearningRate(value)}
        />
      </div>

      <div className="p-2">
        <InputBox
          placeholder="Epochs: 5000"
          onChange={(value) => setEpochs(value)}
        />
      </div>

      <div className="p-4">
        <Button onClick={fetchImage} />
      </div>

      {loading && <p>Loading image...</p>}

    </div>
  );
}
