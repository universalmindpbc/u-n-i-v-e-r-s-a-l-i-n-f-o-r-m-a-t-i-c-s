<?php
declare(strict_types=1);
header('Content-Type: application/json; charset=utf-8');
header('Cache-Control: no-store');
header('X-Content-Type-Options: nosniff');
function reply(array $body, int $status=200): never {http_response_code($status);echo json_encode($body);exit;}
$host=strtolower(explode(':',$_SERVER['HTTP_HOST']??'')[0]);
if(!in_array($host,['staging.universalmind.tv','universalmind.tv'],true))reply(['error'=>'Open Quantum Atlas to start a research session.'],404);
if($_SERVER['REQUEST_METHOD']!=='POST'||($_SERVER['HTTP_ORIGIN']??'')!=='https://'.$host)reply(['error'=>'Open Quantum Atlas to start a research session.'],403);
$private=rtrim($_SERVER['DOCUMENT_ROOT'],'/').'/vogue/iam/api/.runtime-private';
$config=json_decode((string)@file_get_contents($private.'/informatics-astra.json'),true);
if(!is_string($config['secret']??null)||strlen($config['secret'])<48)reply(['error'=>'The research session service is unavailable. Please retry.'],503);
$body=json_decode(file_get_contents('php://input',false,null,0,4096),true);
$old=$body['previous']??'';$sid=null;$renewUntil=time()+30*86400;
$base64=fn(string $v)=>rtrim(strtr(base64_encode($v),'+/','-_'),'=');
if(is_string($old)&&strlen($old)<2048){
 $parts=explode('.',$old);
 if(count($parts)===2&&hash_equals($base64(hash_hmac('sha256',$parts[0],$config['secret'],true)),$parts[1])){
  $claims=json_decode((string)base64_decode(strtr($parts[0],'-_','+/')),true);
  if(($claims['scope']??'')==='astra-research'&&($claims['aud']??'')==='https://'.$host&&preg_match('/^[a-f0-9]{32}$/',$claims['sid']??'')&&($claims['renewUntil']??0)>time()){$sid=$claims['sid'];$renewUntil=$claims['renewUntil'];}
 }
}
if($sid===null){
 $path=$private.'/astra-session-'.hash('sha256',$_SERVER['REMOTE_ADDR']??'unknown').'.json';
 $f=@fopen($path,'c+');if(!$f||!flock($f,LOCK_EX))reply(['error'=>'Research session storage is unavailable.'],503);
 @chmod($path,0600);$record=json_decode(stream_get_contents($f),true)??['start'=>time(),'count'=>0];
 if(time()-($record['start']??0)>3600)$record=['start'=>time(),'count'=>0];
 if($record['count']>=30){flock($f,LOCK_UN);fclose($f);reply(['error'=>'Research session capacity is temporarily full. Please retry later.'],429);}
 $record['count']++;rewind($f);ftruncate($f,0);fwrite($f,json_encode($record));fflush($f);flock($f,LOCK_UN);fclose($f);$sid=bin2hex(random_bytes(16));
}
$payload=$base64(json_encode(['scope'=>'astra-research','sid'=>$sid,'aud'=>'https://'.$host,'exp'=>time()+43200,'renewUntil'=>$renewUntil]));
reply(['token'=>$payload.'.'.$base64(hash_hmac('sha256',$payload,$config['secret'],true)),'expiresAt'=>gmdate('c',time()+43200)]);
