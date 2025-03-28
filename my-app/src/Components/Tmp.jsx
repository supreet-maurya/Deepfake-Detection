import React from 'react'
import myImage from '../images/img.webp';
import VideoUpload from './VideoUpload';

const Tmp = () => {
  return (
    <>
    
    <div className='bg-gray-500 items-center justify-center p-5 min-h-screen'>

        <h1 className='text-4xl'>Deepfake Detection Software</h1>

        <img src={myImage} className='max-w-3xl max-h-3xl float-right' alt="Placeholder Image" />

        <div className='mt-24 ml-7 my-8 bg-gray-400 w-[40%] min-h-[60vh] flex items-center'>
          <VideoUpload className='relative bottom-0'/>
        </div>
        
    </div>
    
    </>
  )
}

export default Tmp
