// Direct links resolve into the same persistent Informatics frame.
(()=>{
 if(window.parent!==window)return;
 const script=document.currentScript,area=script.dataset.area,url=new URL('./',script.src),source=new URL(location.href);
 url.searchParams.set('entry','research');url.searchParams.set('area',area);
 source.searchParams.delete('embedded');
 if(area==='geometry'){for(const key of['record','record-mode'])if(source.searchParams.has(key))url.searchParams.set(key,source.searchParams.get(key));url.hash=source.hash;}
 else if(source.search||source.hash)url.searchParams.set('section',JSON.stringify({area,search:source.search,hash:source.hash}));
 location.replace(url.href);
})();
