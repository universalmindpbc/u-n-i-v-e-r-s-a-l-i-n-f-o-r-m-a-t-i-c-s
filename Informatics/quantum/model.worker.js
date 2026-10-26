import {verifiedSolve} from './model.mjs';
self.onmessage=({data})=>{
 try{self.postMessage({id:data.id,result:verifiedSolve(data.spec)});}
 catch(error){self.postMessage({id:data.id,error:error.message});}
};
