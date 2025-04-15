# 알고리즘 공부 정리


# 안정 정렬 vs 불안정 정렬


## 안정 정렬 (Stable Sort)
- 안정 정렬은 중복된 값을 입력 순서와 동일하게 정렬하는 정렬 알고리즘
- ex) 삽입 정렬, 병합 정렬, 버블 정렬

## 불안정 정렬 (Unstable Sort)
- 중복된 값이 입력 순서와 동일하지 않게 정렬되는 알고리즘
- ex) 퀵 정렬, 선택 정렬, 계수 정렬


# 정렬 종류

## 버블 정렬 (Bubble Sort)

## 계수 정렬 (Counting Sort)

## 힙 정렬 (Heap Sort)

## 삽입 정렬 (Insertion Sort)

## 합병 정렬 (Merge Sort)

## 퀵 정렬 (Quick Sort)

## 기수 정렬 (Radix Sort)

## 선택 정렬 (Selection Sort)

## 위상 정렬(Topological Sort)
- 선후 관계가 정의된 그래프 구조(순서가 정해진 작업)에서 정렬을 하기 위해 사용
- 끝난 후 진입차수가 0이 아닌 노드가 있다면 순환이 존재한다는 것
- 시간 복잡도 : O(V + E)

1) 그래프의 각 노드들의 진입 차수 테이블 생성 및 진입 차수 계산
2) 진입 차수가 0인 노드를 큐에 넣기
3) 큐에서 노드를 꺼내고 꺼낸 노드 출력
4) 꺼낸 노드와 간선으로 연결된 노드들의 진입 차수--
5) 진입 차수가 0인 노드가 있다면 큐에 넣기
6) 3~5를 큐에 더 이상 아무것도 없을 때까지 반복

- 코드 템플릿

```java
import java.util.LinkedList;

class Topological {
  ArrayList<Integer>[] graph = new ArrayList[N+1];
  Queue<Integer> q = new LinkedList<>();
  int[] edgeCount = new int[9];
  
  public void topological() {
    
    for(int i = 1; i < N; i++) {
        if(edgeCount[i] == 0) {
            q.offer(i);
        }
    }
    
    while(!q.isEmpty()) {
        int nodeNo = q.poll();
        
        List<Integer> list = graph.get(nodeNo);
        
        for(int i = 0; i < list.size(); i++) {
            if(--edgeCount[list.get(i)] == 0) {
                q.offer(list.get(i));
            }
        }
    }
  }
}
```

# DFS vs BFS
- 너무 쉽다 이건


# 이분탐색
- 시간복잡도 : O(logN)

# 유니온-파인드(Union Find)
- 그래프 알고리즘으로 두 노드가 같은 그래프에 속하는지 판별하는 알고리즘.
- 서로소 집합, 상호 베타적 집합, 분리 집합으로도 불림
- union(x,y) 연산과 find(x) 연산으로 이루어져 있다.
- union(x,y) 연산은 x와 y의 그래프를 합친다는 의미이며, find(x)는 x가 어느 그래프에 속하는지 연산한다는 의미이다.
- 일반적으로 O(logN), 최대 O(N), 경로 압축시 O(a(N)). a(N)은 아커만 함수를 의미 : 상수 시간 복잡도로 생각해도 됨

- union(x,y)
  1) x와 y의 부모노드를 찾는다.
  2) 둘이 같은 그래프라면 false를 반환하고
  3) x가 y보다 작다면 arr[y] = x, 반대라면 arr[x] = y

- find(x)
  1) parent[x] == x라면 x를 반환하고
  2) 아니라면 find(parent[x]);
     - 이 과정에서 parent[x] = find(parent[x])를 반환하면 경로 압축을 할 수 있다.

- 코드 템플릿
```java
class UnionFind {
    
    boolean union(int x, int y) {
        x = find(x);
        y = find(y);
        
        if(x == y) {
            return false;
        }
        if(x < y) {
            arr[x] = y;
        } else {
            arr[y] = x;
        }
        
        return true;
    }
    
    int find(int x) {
        if(x == arr[x]) {
            return arr[x];
        }
        return arr[x] = find(arr[x]);
    }
}
```

# 그래프 최단 거리 구하는 알고리즘들
## 밸만-포드 알고리즘
- 그래프 최단 경로 구하는 알고리즘
- 하나의 정점에서 출발하는 최단 거리를 구하며 음수 사이클이 없어야 함.(음수 가중치는 허용)
- 다익스트라에 비해 조금 더 시간복잡도가 증가한 대신 음수 가중치를 처리할 수 있음.
- 시간복잡도 : O(EV)

1) 출발 노드는 0 나머지는 INF로 초기화
2) 간선 m개를 하나씩 살펴보며 dist[v]가 무한대가 아닐 때 dist[v] = min(dist[v], dist[w] + cost(w,v))
3) 1)~2)과정을 모든 노드간 반복

- 음수 사이클 확인하기
  - n개 만큼 반복하는 과정을 한번 더 했을 때 바뀌는 값이 있다면 음수 사이클이 존재하는 것.

- 코드 템플릿
```java
class BellmanFord {
    class Edge {
        int v; //나가는 노드
        int w; //들어오는 노드
        int cost;
        
        public Edge(int v, int w, int cost) {
            this.v = v;
            this.w = w;
            this.cost = cost;
        }
    }
    
    static ArrayList<Edge> graph;
    
    public boolean BellmanFord(int n, int m, int start) {
        int[] dist = new int[n + 1];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[start] = 0;
        
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                Edge edge = graph.get(j);
                if(dist[edge.v] != Integer.MAX_VALUE) {
                    dist[edge.w] = Math.min(dist[edge.w], dist[edge.v] + edge.cost);
                }
            }
        }

        //음수 사이클 검증
        for(int j = 0; j < m; j++) {
            Edge edge = list.get(j);
            if(dist[edge.v] != Integer.MAX_VALUE && dist[edge.w] > dist[edge.v] + edge.cost) {
                System.out.println("음수");
                return;
            }
        }
    }
}
```
## 플로이드-워셜 알고리즘
- 음수 사이클이 없는 그래프 내의 모든 정점에서 각 모든 정점에 까지의 최단거리를 모두 구할 수 있는 알고리즘.
- DP랑, 인접 행렬 이용
- 모든 노드에서 모든 노드로 가는 최소 비용을 단계적으로 갱신하면서 진행되는 알고리즘
- 시간복잡도 : O(N ^ 3)

- 코드 템플릿
```java
public class FloydWarshall {
    
    public static void floydWarshall() {
        int[][] D = new int[N][N];
        
        //갈 수 없는 경로 확인
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                if(i == j) continue;
                if(D[i][j] == 0) D[i][j] = Integer.MAX_VALUE;
            }
        }
        
        //플로이드 워셜
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                if(i == j) continue; //출발지와 경유지가 같으면 다음 탐색
                
                for(int k = 0; k < N; k++) {
                    if(i == k || j == k) continue; //출발지와 도착지가 같거나 도착지가 경유지면 다음 탐색
                    D[j][k] = Math.min(D[j][i] + D[i][k], D[j][k]); // 경유하거나 직접가거나 더 짧은 경로로 대체
                }
            }
        }
    }
}
```

## 다익스트라 알고리즘
- 음의 가중치가 없는 그래프의 한 노드에서 각 모든 노드까지의 최단거리를 구하는 알고리즘을 말함.
- 기본적으로 그리디 + 다이나믹 프로그래밍 기법
- 시간복잡도 : O(ElogV)

1) 아직 방문하지 않은 정점 중 출발지로부터 가장 거리가 짧은 정점을 방문
2) 해당 정점을 거쳐 갈 수 있는 정점의 거리가 이전 기록한 값보다 적으면 갱신

- 코드 템플릿

```java
import java.util.PriorityQueue;

class Dijkstra {
    class Node implements Comparable<Node> {
        int index;
        int cost;

        public Node(int index, int cost) {
            this.index = index;
            this.cost = cost;
        }

        @Override
        public int compareTo(Node o) {
            return Integer.compare(this.cost, o.cost);
        }
    }

    static ArrayList<Node>[] graph;

    public void Dijkstra(int n, int start) {
        boolean[] check = new boolean[n + 1];
        int[] dist = new int[n + 1];
        final int INF = Integer.MAX_VALUE;
        Arrays.fill(dist, INF);
        dist[start] = 0;

        PriorityQueue<Node> pq = new PriorityQueue<>();
        pq.offer(new Node(start, 0));
        
        while(!pq.isEmpty()) {
            int nowVertex = pq.poll().index;
            
            if(check[nowVertex]) continue;
            check[nowVertex] = true;
            
            for(Node next : graph[nowVertex]) {
                if(dist[next.index] > dist[nowVertex] + next.cost) {
                    dist[next.index] = Math.min(dist[next.index], dist[nowVertex] + next.cost);
                    pq.offer(new Node(next.index, dist[next.index]));
                }
            }
        }
    }
}
```


# LCS(Longest Common Subsequence) : 최장 부분 수열 알고리즘
- dp를 응용한 최장 부분 수열 알고리즘
- 시간복잡도 : O(nm)

1) arr을 2차원 배열(arr)로 선언
2) 문자를 하나씩 비교
   1) 일치하는 문자가 나오면 arr[i][j] = arr[i-1][j-1] + 1
   2) 일치하지 않는 문자가 나온다면 arr[i][j] = max(arr[i-1][j], arr[i][j-1])
3) 배열이 끝날때까지 반복

- LCS 값을 구하는 경우 이를 역추적
1) arr[i-1][j] or arr[i][j-1] 중 같은 값이 있다면 거기로 이동
2) 없다면 arr[i-1][j-1]로 이동하면서 arr[i][j] 값을 answer 문자열 앞에 더하기
3) 반복이 끝나면 answer 출력

- 코드 템플릿
```java
public class LCS {
    static int LCS(int N) {
        String s1;
        String s2;
        int[][] dp = new int[N][N];
        
        for(int i = 1; i < N; i++) {
            for(int j = 1; j < N; j++) {
                if(s1.charAt(i-1) == s2.charAt(j-1)) {
                    dp[i][j] = dp[i-1][j-1] + 1;
                }
                else {
                    dp[i][j] = Math.max(dp[i][j-1], dp[i-1][j]);
                }
            }
        }
    }
}

```

# LIS(Longest Increasing Subsequence) : 최장 증가 부분 수열
- 어떠한 수열에서 오름차순으로 증가하는 가장 긴 부분수열을 찾는 것

## dp를 이용한 방법
   - 정확한 LIS를 구할 때 사용
   - 시간 복잡도 : O(N ^ 2)
   1) k의 앞 순서에 있는 모든 원소들 중 값이 k보다 작은 원소에 대해, 각각의 원소에서 끝나는 최장 증가 수열의 길이를 알고 있으면 구할 수 있음.
   2) dp[i] = i번째 인덱스에서 끝나는 최장 증가 수열의 길이

## 이분 탐색을 이용한 방법
   - 정확한 LIS가 아닌 길이만 구할 때 사용
   - 시간 복잡도 : O(NlogN)
   1) 배열을 하나 더 두고 원래 수열에서의 각 원소에 대해 LIS 배열 내에서의 위치를 찾음.
   2) 기존 배열을 탐색 
   2) 현재 수가 배열의 마지막 값보다 크다면 LIS 내 배열에 추가
   3) 현재 수가 배열의 마지막 값보다 작다면 이분 탐색을 통해 들어갈 위치를 찾고 교체.
      1) 이분 탐색 시, start = 0, end = 현재 위치이며 start < end동안 반복 -> end 값으로 교체

- 코드 템플릿
```java
// 이분 탐색으로 길이만 구함
public class LIS {
    static void LIS(int N) {
        int[] arr = new int[N];
        int[] lis = new int[N];
        int current = 0;
        for(int i = 0; i < N; i++) {
            int target = arr[0];
            if(i == 0) {
                lis[0] = target;
                continue;
            } 
            if(target > lis[current]) {
                lis[++current] = target;
            } else {
                int start = 0;
                int end = current;
                while(start < end) {
                    int mid = (start + end) >>> 1;
                    if(lis[mid] < target) {
                        start = mid + 1;
                    } else {
                        end = mid;
                    }
                }
                lis[end] = target;
            }
        }
    }
}
```
# 투 포인터

# MST(Minimum Spanning Tree)
- Spanning Tree란?
  - 그래프 내의 모든 정점을 포함하는 트리
    - 최소 연결 = 간선의 수가 가정 적다
    - n개의 정점을 가지는 그래프의 최소 간선의 수는 (n-1)개이고, (n-1)개의 간선으로 연결되어 있으면 필연적으로 트리 형태가 되고 이것이 Spanning Tree
- DFS, BFS을 이용하여 그래프에서 신장 트리를 찾을 수 있음.
- 모든 정점들이 연결되어야 하고, 사이클을 포함해서는 안됨.

- MST(Minimum Spanning Tree)란?
  - Spanning Tree 중에서 사용된 간선들의 가중치 합이 최소인 트리
  - 네트워크에 있는 모든 정점들을 가장 적은 수의 간선과 비용으로 연결하는 것
- 특징
  - 간선의 가중치의 합이 최소여야 함
  - n개의 정점을 가지는 그래프에 대해 반드시 (n-1)개의 간선만을 사용해야 함
  - 사이클이 포함되어서는 안됨
- 구현 방법
  1) Kruskal MST 알고리즘
    - 그리디 알고리즘을 이용하여 네트워크의 모든 정점을 최소 비용으로 연결하는 최적 해답을 구하는 법
    - Union-Find 알고리즘을 활용하여 구현
    - 시간 복잡도 : O(ElogE)
  
    - 동작 방식
      1) 그래프의 간선들을 가중치 기준 오름차순으로 정렬
      2) 정렬된 간선 리스트를 순서대로 선택, 간선의 정점들을 연결
      3) 정점을 연결하는 것은 Union-Find의 Union으로 구현
      4) 간선의 두 정점 a,b가 이미 연결되어 있다면 스킵
      5) 위 과정을 반복하여 최소 비용의 간선들만 이용하여 모든 정점이 연결됨
  2) Prim MST 알고리즘
  - 시작 정점에서부터 출발하여 신장트리 집합을 단계적으로 확장해나가는 방법
  - 우선순위 큐를 활용하여 구현
  - 시간복잡도 : O(ElogV)

  - 동작 방식
    1) 임의의 정점을 시작점으로 선택
    2) 갈 수 있는 정점 중 가장 가중치가 작은 정점 연결
    3) 새로운 정점 중 가중치가 가장 작은 정점으로 연결

  - 코드 템플릿

```java
  import java.util.PriorityQueue;

class Prim {
    static class Edge implements Comparable<Edge> {
        Edge(int w, int cost) {
            this.w = w;
            this.cost = cost;
        }

        @Override
        public int compareTo(Edge o) {
            return this.cost - o.cost;
        }
    }
    
    static List<Edge>[] graph;

    public static void prim(int start, int n) {
        boolean[] visit = new boolean[n + 1];

        PriorityQueue<Edge> pq = new PriorityQueue<>();
        pq.offer(new Edge(start, 0));
        
        int total = 0;
        while(!pq.isEmpty()) {
            Edge edge = pq.poll();
            
            int v = edge.w;
            int cost = edge.cost;
            
            if(visit[v]) continue;
            
            visit[v] = true;
            total += cost;
            
            for(Edge e : graph[v]) {
                if(!visit[e.w]) {
                    pq.add(e);
                }
            }
        }
    }
}
  ```
# Bit Masking

# DP(Dynamic Programming)
- DP란?
  - 하나의 큰 문제를 작은 문제로 나누어 해결하는 기법
- 구현 방법
  1) 먼저 DP로 풀 수 있는 문제인지 확인.
  2) 각 소문제들을 나누는 변수를 정함.
  3) 각 변수들간의 관계식, 즉 점화식을 찾기.
  - Top-Down
    - 주로 재귀함수를 사용하며 dp[n]값을 찾기 위한 재귀 함수의 호출이 dp[0]까지 내려간 다음 결과들이 재귀 함수에 맞물리며 재활용되는 방식.
    ```
    dp = Math.min(재귀함수 경로1() + value, 재귀함수 경로2() + value);
    ```
  - Bottom-Up
    - 초기조건을 기반으로 차곡차곡 데이터를 쌓아가 큰 문제의 결과를 도출하는 과정
- 정형화된 코드 템플릿은 없는 듯하다.
## DP 예시
- 계단
- 냅색
- LCS
- 트리 만들기
- 블록 채우기

# 냅색(Knapsack)
-  Knapsack Problem, 배낭문제는 다이나믹 프로그래밍에서 매우 유명한 문제
- 냅색이란?
  - 어떤 배낭이 있고 그 배낭안에 넣을 수 있는 최대 무게가 K라고 하자. 배낭에 넣을 수 있는 N개의 물건이 각기 다른 가치 V를 가지고 있고 각 물건마다 다른 무게 W를 가지고 있을 때, 배낭이 최대한 가치가 높은 물건들을 담을 수 있는 조합을 찾는 문제이다.
  - 해당 문제는 물건을 쪼갤 수 있는 Fraction Knapsack Problem과 물건을 쪼갤 수 없는 0-1 knapSack Problem으로 나뉜다.
- [0-1 Knapsack Problem](https://howudong.tistory.com/106)
- dp[i][j]는 i번째 물건까지의 각 무게별 가치 최대값
- 물건 K의 무게 > 배낭 W 무게
  - dp [K][W] = dp [K-1][W]
- 물건 K의 무게 <= 배낭 W 무게
  - dp [K][W] = max(dp [K-1][W], K가치 + dp [K-1][W-K무게])

# 비트 필드(Bit Field)를 이용한 다이나믹 프로그래밍
- 방문 배열을 응용한 비트 필드
- dp의 인덱스를 비트마스킹으로 관리하자.
- 응용이 무궁무진하니 다양한 방법을 보고 익히자

```java
import java.util.*;
import java.io.*;

class BitFieldDP {
	
	static int[] dp;
	static int[][] arr;
	static int N;
	static final int DEFAULT = Integer.MAX_VALUE >> 1;
	public static void main(String args[]) throws Exception {

		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		
		N = Integer.parseInt(br.readLine());
		arr = new int[N][N];
		dp = new int[1 << N];
		
		for(int i = 0; i < N; i++) {
			StringTokenizer st = new StringTokenizer(br.readLine());
			for(int j = 0; j < N; j++) {
				arr[i][j] = Integer.parseInt(st.nextToken());
			}
		}
		
		Arrays.fill(dp, DEFAULT);
		
		System.out.println(DP(N, (1 << N) - 1));
		
	}
	
	public static int DP(int n, int k) {
		if(dp[k] != DEFAULT) return dp[k];
		
		for(int i = 0; i < N; i++) {
			if((k & (1 << i)) == 0) continue;
			
			int next = k ^ (1 << i);
			
			if(dp[k] == DEFAULT) dp[k] = DP(n - 1, next) + arr[n - 1][i];
			else dp[k] = Math.min(dp[k], DP(n - 1, next) + arr[n - 1][i]);
		}
		return dp[k];
	}
	
}
```
- 정해(재귀를 이용한 방법)
```java
    private static int tsp(int city, int visited) {
        if (visited == (1 << n) - 1)
            return distance[city][0];
        if (dp[city][visited] != -1)
            return dp[city][visited];
        int result = Integer.MAX_VALUE;
        for (int nextCity = 0; nextCity < n; nextCity++) {
            if ((visited & (1 << nextCity)) != 0)
                continue;
            int temp = distance[city][nextCity] + tsp(nextCity, visited | (1 << nextCity));
            result = Math.min(result, temp);
        }
        dp[city][visited] = result;
        return result;
    }
```
# 세그먼트 트리
- 어쩔 세그~
```java
import java.io.BufferedWriter;
import java.io.OutputStreamWriter;
import java.util.Objects;

public class Main {
  static Node[] A;
  static int size;
  public static void main(String[] args) throws Exception {
    int N =  read();
    BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));
    size = 1;
    while(size < N) {
      size <<= 1;
    }
    A = new Node[size * 2 + 1];
    for(int i = 0; i < size * 2 + 1; i++) {
      A[i] = new Node(Integer.MAX_VALUE,Integer.MAX_VALUE);
    }
    for(int i = size + 1; i < size + N + 1; i++) {
      A[i] = new Node(i - size, read());
    }
    update();

    int M =  read();
    while(M --> 0) {
      int a =  read();
      if(a == 1) {
        updateValue(new Node(read(), read()));
        continue;
      }
      int i =  read();
      int j =  read();
      bw.write(query(i, j, 2, 1, size).index + "\n");
    }
    bw.flush();

  }

  private static Node query(int left, int right, int node, int start, int end) {
    if(left > end || right < start) {
      return new Node(Integer.MAX_VALUE, Integer.MAX_VALUE);
    }
    if(left <= start && end <= right) {
      return A[node];
    }
    int mid = (start + end) / 2;
    Node n1 = query(left, right, node * 2 - 1, start, mid);
    Node n2 = query(left, right, node * 2, mid + 1, end);
    if(n1.compareTo(n2) > 0) {
      return n1;
    }
    return n2;
  }

  private static void updateValue(Node node) {
    int index = size + node.index;
    A[index].val = node.val;

    while(index > 1) {
      int current = (index + 1) >> 1;

      //이거 고치기
      Node n1 = A[(current << 1) -1];
      Node n2 =  A[current << 1];
      if(n1.compareTo(n2) > 0) {
        A[current] = n1;

      } else {
        A[current] = n2;
      }
      index = current;
    }
  }

  private static void update() {
    int size = A.length - 1;
    while(size > 1) {
      Node n1 =  A[size-1];
      Node n2 = A[size];
      if(n1.compareTo(n2) > 0) {
        A[(size + 1) >> 1] = n1;
      } else {
        A[(size + 1) >> 1] = n2;
      }
      size -= 2;
    }
  }

  private static class Node implements Comparable<Node> {
    int val;
    int index;

    public Node(int index, int val) {
      this.index = index;
      this.val = val;
    }

    @Override
    public int compareTo(Node o) {
      if(o.val == this.val) {
        return o.index - this.index;
      }
      return  o.val -  this.val;
    }
  }
  
  private static int read() throws Exception {
    int d, o;
    boolean negative = false;
    d = System.in.read();

    if (d == '-') {
      negative = true;
      d = System.in.read();
    }
    o = d & 15;
    while ((d = System.in.read()) > 32)
      o = (o << 3) + (o << 1) + (d & 15);

    return negative? -o:o;
  }

}


```

# 느리게(lazy) 갱신되는 세그먼트 트리
- 어쩔 세그~
- 코드 템플릿
```java
import java.io.*;

public class Main {
  static long[] segment;
  static long[] lazy;
  static int size;
  public static void main(String[] args) throws Exception {
    int N = read();
    BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));
    size = 1;
    while(size < N) {
      size <<= 1;
    }

    segment = new long[(size << 1) + 1];
    lazy = new long[(size << 1) + 1];

    for(int i = size + 1; i < size + N + 1; i++) {
      segment[i] = read();
    }

    int segmentSize = segment.length - 1;

    while(segmentSize > 1) {
      segment[(segmentSize + 1) >> 1] = segment[segmentSize] + segment[segmentSize - 1];
      segmentSize-=2;
    }

    int M = read();

    for(int i = 0; i < M ;i++) {
      int a = read();
      if(a == 2) {
        int node = read();
        query(node, node, 2, 1, size);
        bw.write(segment[size + node] + "\n");
        continue;
      }
      updateRange(read(), read(), 2, 1, size, read());
    }

    bw.flush();

  }

  private static void updateRange(int left, int right, int node, int start, int end, int diff) {
    if(right < start || left > end) {
      return;
    }

    if(left <= start && end <= right) {
      lazy[node] += diff;
      int segmentSize = (node + 1) >> 1;
      while(segmentSize > 1) {
        segment[segmentSize] += (long) diff * (end - start + 1);
        segmentSize = (segmentSize + 1) >> 1;
      }
      return;
    }
    int mid = (start + end) / 2;
    updateRange(left, right, node * 2 - 1, start, mid, diff);
    updateRange(left, right, node * 2, mid+1, end, diff);
  }

  private static void query(int left, int right, int node, int start, int end) {
    if(lazy[node] != 0) {
      segment[node] += lazy[node] * (end - start + 1);
      updateLazy(node);
    }
    if(right < start || left > end) {
      return;
    }

    if(left <= start && end <= right) {
      return;
    }
    int mid = (start + end) / 2;
    query(left, right, node * 2 - 1, start, mid);
    query(left, right, node * 2, mid+1, end);
  }

  private static void updateLazy(int i) {
    long tmp = lazy[i];
    lazy[i] = 0;
    if(size < i) return;
    lazy[i * 2] += tmp;
    lazy[i * 2 - 1] += tmp;
  }


  private static int read() throws Exception {
    int d, o;
    boolean negative = false;
    d = System.in.read();

    if (d == '-') {
      negative = true;
      d = System.in.read();
    }
    o = d & 15;
    while ((d = System.in.read()) > 32)
      o = (o << 3) + (o << 1) + (d & 15);

    return negative? -o:o;
  }

}

```

# 강한 결합 요소 - SCC (Strong Connected Component)
- `강하게 결합된 정점 집합`을 의미
- 서로 긴밀하게 연결되어 있다고 하여 강한 결합 요솧라고 함.
- 특징
  - 같은 SCC에 속하는 두 정점은 서로 도달이 가능하다는 특징이 있음.
  - 사이클이 발생하는 경우 무조건 SCC에 해당
  - 무향 그래프라면 그 그래프는 무조건 SCC
  - 일반적으로 코사라주 알고리즘이 더 구현은 쉽지만 타잔 알고리즘이 더 적용이 쉬움.

## 타잔 알고리즘(Tarjan's Algorithm)
- 순방향 간선, 역방향 간선, 교차 간선을 다 고려해야 됨
- 모든 정점에 대해 ALL DFS(모든 정점에서 수행되는 DFS) 한번으로 모든 SCC를 구하는 알고리즘.
- 부모에서 자식으로 나아가는 알고리즘으로 부모로 다시 돌아올 수 있는 경로에 한해 SCC가 성립됨.
- 타잔 알고리즘은 위상 정렬을 이용한 방법으로 생성되는 SCC들은 위상 정렬의 역순으로 생성된다.
- 시간복잡도 : O(V + E)
- 구현 방법
  1. ALL DFS를 돌리며 Spanning Tree를 만들어 갈 때 DFS의 호출 순서에 따라 정점을 stack에 push
  2. 간선 분류를 통해 먼저 호출되는 정점이 더 높은 위치를 가진다고 생각할 때 가장 높이 올라갈 수 있는 정점을 찾음.
  3. 이때 here -> there이 교차 간선이지만 there이 아직 SCC에 속하지 않는다면 discover[there]을 고려해줌.
  4. DFS가 끝나기 전에 ret과 discover[here]가 같다면 stack에서 pop하면서 here가 나올 때까지 같은 SCC로 분류함.
[참고](https://m.blog.naver.com/PostView.nhn?blogId=kks227&logNo=220802519976&referrerCode=0&searchKeyword=scc)

## 코사라주 알고리즘(Kosaraju's Algorithm)

- DFS를 정방향으로, 역방향으로 한번씩 하여 구현하는 알고리즘.
- 시간복잡도 : O(V + E)
- 구현 방법
  1. 모든 정점에 대해 정방향 그래프를 DFS로 수행하며 끝나는 순서대로 스택에 삽입
     1. 방문하지 않은 정점이 있는 경우에는 해당 정점부터 다시 DFS를 수행한다.
     2. 이로써 모든 정점을 방문하며 DFS를 완료하여, 스택에 모든 정점을 담는다.
  2. 스택의 top에서부터 pop()을 진행하며 순서대로 역방향 그래프에서 DFS를 수행하며 한번 수행에 탐색되는 모든 정점들을 같은 SCC로 묶음. 
     1. 이 과정은 스택이 빌 때까지 진행
     2. 만약 스택의 top에 위치한 정점이 이미 방문되었다면 pop()만 함.
```java
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Stack;

public class Main {


    static Stack<Integer> stack = new Stack<>();
    static ArrayList<Integer>[] list;
    static ArrayList<Integer>[] reverseList;
    static ArrayList<ArrayList<Integer>> answer;
    static ArrayList<Integer> ans;
    static boolean[] visited;

    //코사라주 알고리즘
    public static void main(String[] args) throws Exception {
        int V = read();
        int E = read();

        list = new ArrayList[V+1];
        reverseList = new ArrayList[V + 1];
        visited = new boolean[V+1];
        answer = new ArrayList<>();

        for(int i = 0; i <= V; i++) {
            list[i] = new ArrayList<>();
            reverseList[i] = new ArrayList<>();
        }

        for(int i = 0; i < E; i++) {
            int a = read();
            int b = read();
            list[a].add(b);
            reverseList[b].add(a);
        }

        // SCC
        for(int i = 1; i <= V; i++) {
            if(visited[i]) continue;
            visited[i] = true;
            dfs(i);
        }

        visited = new boolean[V+1];

        while(!stack.isEmpty()) {
            int i = stack.pop();
            if(visited[i]) continue;
            ans = new ArrayList<>();
            visited[i] = true;
            reverseDfs(i);
            answer.add(ans);
        }
        StringBuilder sb = new StringBuilder();
        System.out.println(answer.size());

        for (ArrayList<Integer> ans : answer) Collections.sort(ans);
        answer.sort(Comparator.comparingInt(o1 -> o1.get(0)));

        for (ArrayList<Integer> ans : answer) {
            for (int j : ans) sb.append(j).append(" ");
            sb.append("-1\n");
        }

        System.out.println(sb);

    }

    public static void reverseDfs(int i) {
        ans.add(i);
        for(int j : reverseList[i]) {
            if(visited[j]) continue;
            visited[j] = true;
            reverseDfs(j);
        }
    }

    public static void dfs(int i) {
        for(int j : list[i]) {
            if(visited[j]) continue;
            visited[j] = true;
            dfs(j);
        }
        stack.push(i);
    }

    private static int read() throws Exception {
        int d, o;
        boolean negative = false;
        d = System.in.read();

        if (d == '-') {
            negative = true;
            d = System.in.read();
        }
        o = d & 15;
        while ((d = System.in.read()) > 32)
            o = (o << 3) + (o << 1) + (d & 15);

        return negative? -o:o;
    }

}

```

# 최소 공통 조상 - LCA(Lowest Common Ancestor)

# 볼록 껍질 알고리즘

# CCW(Counter Clock Wise)
- 3개의 점 r, p, q가 있을 때 벡터 rp를 기준으로 점 q가 어느 위치(왼쪽, 같은 직선, 오른쪽)에 있는지 판별하는 방법
- 벡터의 외적
- 왼쪽에 있을 때(반시계 방향일 때) : 1
- 같은 직선상에 있을 때(직선일 때) : 0
- 오른쪽에 있을 때(시계 방향일 때) : -1

## 외적
- 두 벡터 사이의 곱 연산
- 외적의 결과물은 두 벡터가 사이에 이루는 평행 사변형의 면적
- 교환 법칙이 성립하지 않음.
- 오른손 법칙이 성립
- 신발끈 공식을 사용하여 벡터의 곱을 판단.

## 신발끈 공식
- 볼록 및 오목 다각형의 넓이를 구하는 공식
- 반시계 방향(혹은 시계 방향)으로 정렬된 점들로 이루어진 넓이를 구함
- ![image](https://github.com/user-attachments/assets/6a46c546-24d9-4dcd-8622-1c491ee26cd5)
![image](https://github.com/user-attachments/assets/19edab0c-eb5b-4a87-8d2d-3f59ee63e221)

- 여기서는 세 점의 방향을 구해야 하므로 `(x1y2 + x2y3 + x3y1) - (x2y1 * x3y2 * x1y3)` 로 구해짐
```java
class CCW {

        public static int ccw(int[][] node) {

            int a = (node[0][0] * node[1][1]) + (node[1][0] * node[2][1]) + (node[2][0] * node[0][1]);
            int b = (node[0][1] * node[1][0]) + (node[1][1] * node[2][0]) + (node[2][1] * node[0][0]);
            return a - b;
        }

```

# KMP 알고리즘
- `Word`와 `Pattern`이 일치하는 지를 빠르게 비교하는 알고리즘
- 핵심은 현재 값과 패턴 값이 틀려졌을 때, 현재 값이 패턴의 어디에 위치해있는 지를 찾아 최적화 시키는 것이다.
- O(N+M) Word의 길이와 Pattern의 길이를 더한 값
``` java
import java.util.*;
import java.io.*;

class KMP
{

	static int[] table;
	static int count;
	
	public static void failureFunction(String pattern) {
		
		// 접두사 인덱스
		int pIdx = 0;
		int length = pattern.length();
		char[] arr = pattern.toCharArray();
		
		// 접미사 인덱스
		for(int idx = 1; idx < length; idx++) {
			
			// 일치하지 않을 경우 일치하는 패턴 찾기
			while(pIdx != 0 && arr[idx] != arr[pIdx]) {
				pIdx = table[pIdx - 1];
			}
				
			// 일치하는 경우 pIdx 증가시켜주기
			if(arr[idx] == arr[pIdx]) {
				pIdx++;
				table[idx] = pIdx;
			}
		}
	}
	
	public static void KMP(String word, String pattern) {
		
		// 접두사 인덱스
		int pIdx = 0;
		// 일치하는지 비교?
		char[] wordArr = word.toCharArray();
		char[] patternArr = pattern.toCharArray();
		for(int i = 0; i < word.length(); i++) {
			// 일치 안하면 접두사 패턴 찾으러 내려가고
			while(pIdx != 0 && patternArr[pIdx] != wordArr[i]) {
				pIdx = table[pIdx - 1];
			}
			// 만약 pIdx가 끝까지 갔으면? 일치 + 1 시켜주고?
			// pIdx = table[pIdx]로 하여 패턴이 일치하는 부분을 보지 않도록 한다
			if(wordArr[i] == patternArr[pIdx]) {
				if(pIdx == pattern.length() - 1) {
					count++;
					pIdx = table[pIdx];
					continue;
				}
				pIdx++;
				
			}
		}
	}
}

```
![20250114_203949](https://github.com/user-attachments/assets/d48a5819-3c27-49a9-8e55-ebc95a36caa4)
- 왜 패턴을 찾을 떄 pIdx = table[pIdx - 1]인가?
- table[i]는 0부터 i번쨰 문자일때 접두사/접미사가 얼마나 겹치는가 이다.
- 만약 table[pIdx - 1], 즉 현재 위치의 값이 초기 접두사 이후 값과 일치하는 지를 비교한다는 것이다.
- 만약 최장이 아닌 접두사와 현재 위치 이전에 접두사 만큼 일치한다면, Pattern의 접두사를 생략하고 Pattern의 PIdx 값과 현재 위치와 비교할 수 있다.
- 그리고 둘이 일치한다는 말은, 겹치는 길이가 최장인 접두사(pIdx)의 최장 접두사(table[pIdx])가 일치한다는 것이고 (pIdx의 접두 접미가 같기 떄문) table[pIdx] + 1 위치부터 살펴불 수 있다는 것이다. (배열 시작을 0이냐 1로 두냐에 따라 1의 오차가 있을 순 있다)
- 이는 접두사 값을 얼마나 살릴 수 있느냐를 의미한다. 즉, 현재 값이 패턴의 어느쯤 와 있는 지 알아내서 해당 부분부터 비교하게 한다.

# 페르마의 소정리 (모듈러의 역원)
## 나눗셈에서도 나머지 연산을 분배 시키는 방법 (서로소이면서 MOD가 소수일 경우 적용)

- 모듈러 연산(%)는 덧셈, 뺄셈, 곱셉의 분배 규칙은 성립하나, 나눗셈에 분배 규칙이 성립하지 않는다.
- 그렇다면 나눗셈을 곱셈으로 바꾸는 역원을 구한 뒤 모듈러 연산을 하면 되지 않을까?
- 그리고 두 수가 서로소이면서 `MOD`가 소수라면 페르마의 소정리가 적용되어, $a^{(p - 2)} ≡ a^{-1} ( mod p)$가 된다.

- [페르마의 소정리 증명](https://m.blog.naver.com/a4gkyum/220768006509)

```java
public static long combination(int n, int r) {
		
		// n! / (n-r)!r! % MOD
		// = (n! / r!) / (n-r)! % MOD 
		// = (n! / r!) % MOD * (Math.pow((n-r)!, -1) % MOD) (모듈러는 곱셈의 분배 법칙이 성립)
		// = (n! / r!) % MOD * (Math.pow((n-r)!, MOD - 2) % MOD) (페르마의 소정리, 모듈러의 역원) 
		r = Math.max(n - r, r);
		long numerator = 1; // 분자
		
		// (n! / r!) % MOD
		for(int i = r + 1; i <= n; i++) {
			numerator = (numerator * i) % MOD;
		}
		
		// (n-r)! % MOD
		long denominator = 1; // 분모
		for(int i = 1; i <= (n-r); i++) {
			denominator = (denominator * i) % MOD;
		}
		
		// pow((n-r)!, MOD - 2) mod MOD = pow(n-r)!, -1) mod MOD (모듈러의 역원)
		denominator = pow(denominator, MOD - 2);
		
		return (numerator * denominator) % MOD;
	}
	
	// a ^ b % MOD
	// 분할 정복..? 이분 계산..?
	public static long pow(long a, long b) {
		
		long result = 1;
		
		while(b > 0) {
			// a ^ b = a * (a ^ (b-1))
			if((b & 1) == 1) {
				result = (a * result) % MOD;
			}
			
			// a ^ b = (a ^ 2) ^ (b/2) (b % 2 == 0)
			a = (a * a) % MOD;
			b >>= 1;
		}
		
		return result;
	}
```
